import argparse
import random
from pathlib import Path

import clip
import imageio.v2 as imageio
import numpy as np
import PIL.Image
import pydiffvg
import torch
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", nargs="*", default=None, help="Text prompt(s). Multiple prompts are cycled like ref images.")
    parser.add_argument("--negative-prompt", nargs="*", default=None, help="Negative text prompt(s) to push away from (applied every step)")
    parser.add_argument("--ref-image", nargs="+", default=None, help="Path(s) to reference image(s). Multiple images are cycled through evenly across iterations.")
    parser.add_argument("--background-image", default=None, help="PNG/JPG background image composited under the strokes during optimization and export")
    parser.add_argument("--prompt-mask", default=None, help="Mask image for prompt CLIP loss")
    parser.add_argument("--ref-mask", default=None, help="Mask image for reference loss")
    parser.add_argument("--ref-weight", type=float, default=1.0, help="Weight for reference image loss relative to text loss")
    parser.add_argument("--ref-weight-curve", choices=["constant", "linear"], default="constant", help="How reference-image weight evolves within each phase")
    parser.add_argument("--ref-loss-type", choices=["clip", "mse"], default="clip", help="Loss to use for reference images")
    parser.add_argument("--resume", default=None, help="Output dir from a previous run to resume from (loads latest.svg and prepends progress frames)")
    parser.add_argument("--output-dir", default="outputs/default")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-width", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--num-augs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0, help="Base learning rate for points (width=lr/10, color=lr/100)")
    parser.add_argument("--use-normalized-clip", action="store_true")
    parser.add_argument("--phase-lr-reset", action="store_true", help="Reset LR schedule per phase so each phase starts with full LR")
    parser.add_argument("--phase-optim-reset", action="store_true", help="Reset Adam optimizer state on phase switch")
    parser.add_argument("--phase-perturb", type=float, default=0.0, help="Std of gaussian noise added to control points on phase switch (in pixels)")
    parser.add_argument("--phase-blend", type=int, default=0, help="Number of steps to linearly blend from old target to new target on switch")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_augment(use_normalized_clip):
    ops = [
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ]
    if use_normalized_clip:
        ops.append(
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        )
    return transforms.Compose(ops)


def init_scene(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for _ in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.extend([p1, p2, p3])
            p0 = p3

        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(1.0),
            is_closed=False,
        )
        shapes.append(path)
        shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor(
                    [random.random(), random.random(), random.random(), random.random()]
                ),
            )
        )
    return shapes, shape_groups


def tensor_to_uint8(img):
    array = img.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def load_canvas_image(path, canvas_width, canvas_height, device):
    image_pil = PIL.Image.open(path).convert("RGBA")
    image_pil = image_pil.resize((canvas_width, canvas_height), PIL.Image.LANCZOS)
    image_array = np.asarray(image_pil).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_array[:, :, :3]).to(device)
    image_alpha = torch.from_numpy(image_array[:, :, 3:4]).to(device)
    white_bg = torch.ones(canvas_height, canvas_width, 3, device=device)
    return image_alpha * image_rgb + (1 - image_alpha) * white_bg


def load_mask_image(path, canvas_width, canvas_height, device):
    mask_pil = PIL.Image.open(path)
    if "A" in mask_pil.getbands():
        mask_pil = mask_pil.getchannel("A")
    else:
        mask_pil = mask_pil.convert("L")
    mask_pil = mask_pil.resize((canvas_width, canvas_height), PIL.Image.LANCZOS)
    mask_array = np.asarray(mask_pil).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask_array).to(device).unsqueeze(-1)
    if torch.count_nonzero(mask) == 0:
        raise ValueError(f"Mask is empty: {path}")
    return mask


def apply_gradient_mask(img, mask):
    if mask is None:
        return img
    return img * mask + img.detach() * (1 - mask)


def masked_mse_loss(img, target, mask):
    if mask is None:
        return torch.mean((img - target) ** 2)
    weight = mask.expand_as(img)
    return torch.sum(weight * (img - target) ** 2) / torch.sum(weight)


def write_gif(frames, path, fps):
    imageio.mimsave(path, frames, format="GIF", duration=1 / fps, loop=0)


def write_mp4(frames, path, fps):
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def render_partial_scene(render, canvas_width, canvas_height, shapes, shape_groups, seed, background):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(canvas_width, canvas_height, 2, 2, seed, None, *scene_args)
    return img[:, :, 3:4] * img[:, :, :3] + background * (1 - img[:, :, 3:4])


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iters_dir = output_dir / "iters"
    iters_dir.mkdir(exist_ok=True)

    pydiffvg.set_print_timing(False)
    pydiffvg.set_use_gpu(True)
    pydiffvg.set_device(device)

    canvas_width = args.canvas_size
    canvas_height = args.canvas_size

    prompts = args.prompt or []
    ref_images = args.ref_image or []
    if not prompts and not ref_images:
        raise ValueError("At least one of --prompt or --ref-image is required.")

    negative_prompts = args.negative_prompt or []

    text_features_list = []
    neg_features_list = []
    ref_features_list = []
    ref_mse_images_list = []
    if prompts or negative_prompts or args.ref_loss_type == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        model.visual = torch.compile(model.visual)

        for p in prompts:
            if p.strip():
                with torch.inference_mode():
                    text_features_list.append(model.encode_text(clip.tokenize(p).to(device)))
                print(f"Encoded prompt: {p!r}")
            else:
                text_features_list.append(None)
                print(f"Blank prompt: {p!r} (ref loss only)")

        for p in negative_prompts:
            with torch.inference_mode():
                neg_features_list.append(model.encode_text(clip.tokenize(p).to(device)))
            print(f"Encoded negative prompt: {p!r}")

        if args.ref_loss_type == "clip":
            for ref_path in ref_images:
                ref_pil = PIL.Image.open(ref_path).convert("RGB")
                ref_tensor = preprocess(ref_pil).unsqueeze(0).to(device)
                with torch.inference_mode():
                    ref_features_list.append(model.encode_image(ref_tensor))
                print(f"Encoded ref image (clip): {ref_path}")
    else:
        model = None

    if args.ref_loss_type == "mse":
        for ref_path in ref_images:
            ref_mse_images_list.append(load_canvas_image(ref_path, canvas_width, canvas_height, device))
            print(f"Loaded ref image (mse): {ref_path}")

    # Determine number of phases and broadcast
    n_text = len(text_features_list)
    n_ref = len(ref_images)
    if n_text > 1 and n_ref > 1:
        assert n_text == n_ref, (
            f"Got {n_text} prompts and {n_ref} ref images — must be equal, "
            "or one side must be 0 or 1."
        )
    n_phases = max(n_text, n_ref, 1)

    def get_text_feat(phase_idx):
        if not text_features_list:
            return None
        return text_features_list[phase_idx % n_text]

    def get_ref_feat(phase_idx):
        if not ref_features_list:
            return None
        return ref_features_list[phase_idx % n_ref]

    def get_ref_mse_image(phase_idx):
        if not ref_mse_images_list:
            return None
        return ref_mse_images_list[phase_idx % n_ref]

    def get_phase_ref_weight(step_in_phase, phase_length):
        if args.ref_weight_curve == "constant":
            return args.ref_weight
        if phase_length <= 1:
            return args.ref_weight
        frac = step_in_phase / (phase_length - 1)
        frac = min(max(frac, 0.0), 1.0)
        return args.ref_weight * frac

    target_text_feat = get_text_feat(0)
    target_ref_feat = get_ref_feat(0)
    target_ref_mse_image = get_ref_mse_image(0)

    if n_phases > 1:
        iters_per_phase = args.num_iter // n_phases
        switch_steps = {iters_per_phase * i: i for i in range(1, n_phases)}
        print(f"Multi-target: {n_phases} phases, switching at steps {sorted(switch_steps)}")
    else:
        switch_steps = {}

    augment = make_augment(args.use_normalized_clip)
    render = pydiffvg.RenderFunction.apply
    progress_frames = []

    if args.resume:
        resume_dir = Path(args.resume)
        resume_svg = resume_dir / "latest.svg"
        _, _, shapes, shape_groups = pydiffvg.svg_to_scene(str(resume_svg))
        print(f"Resumed {len(shapes)} paths from {resume_svg}")

        resume_iters = sorted((resume_dir / "iters").glob("iter_*.png"))
        for p in resume_iters:
            frame = np.array(PIL.Image.open(p).convert("RGB"))
            progress_frames.append(frame)
        print(f"Loaded {len(progress_frames)} prior frames from {resume_dir / 'iters'}")
    else:
        shapes, shape_groups = init_scene(args.num_paths, canvas_width, canvas_height)

    if args.background_image:
        background = load_canvas_image(args.background_image, canvas_width, canvas_height, device)
        print(f"Loaded background image: {args.background_image}")
    else:
        background = torch.ones(canvas_height, canvas_width, 3, device=device)

    if args.prompt_mask:
        prompt_mask = load_mask_image(args.prompt_mask, canvas_width, canvas_height, device)
        print(f"Loaded prompt mask: {args.prompt_mask}")
    else:
        prompt_mask = None

    if args.ref_mask:
        ref_mask = load_mask_image(args.ref_mask, canvas_width, canvas_height, device)
        print(f"Loaded ref mask: {args.ref_mask}")
    else:
        ref_mask = None

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        path.stroke_width.requires_grad = True
        points_vars.append(path.points)
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    optim = torch.optim.Adam([
        {"params": points_vars, "lr": args.lr},
        {"params": stroke_width_vars, "lr": args.lr / 10},
        {"params": color_vars, "lr": args.lr / 100},
    ], foreach=True)

    # Per-phase LR schedule helper
    base_lrs = [g["lr"] for g in optim.param_groups]

    def apply_phase_lr(step_in_phase, phase_length):
        frac = step_in_phase / phase_length if phase_length > 0 else 1.0
        if frac < 0.5:
            scale = 1.0
        elif frac < 0.75:
            scale = 0.4
        else:
            scale = 0.1
        for g, blr in zip(optim.param_groups, base_lrs):
            g["lr"] = blr * scale

    current_phase_idx = 0
    phase_start_step = 0
    current_phase_len = list(switch_steps.keys())[0] if switch_steps else args.num_iter

    # For blending: track previous target
    prev_target_text_feat = None
    prev_target_ref_feat = None
    prev_target_ref_weight = 0.0
    prev_target_ref_mse_image = None
    blend_start_step = -1

    final_img = None
    for step in range(args.num_iter):
        # Global LR schedule (used when phase-lr-reset is off)
        if not args.phase_lr_reset:
            if step == int(args.num_iter * 0.5):
                optim.param_groups[0]["lr"] = 0.4
            if step == int(args.num_iter * 0.75):
                optim.param_groups[0]["lr"] = 0.1
        else:
            apply_phase_lr(step - phase_start_step, current_phase_len)

        if step in switch_steps:
            phase_idx = switch_steps[step]

            if args.phase_blend > 0:
                prev_target_text_feat = None if target_text_feat is None else target_text_feat.clone()
                prev_target_ref_feat = None if target_ref_feat is None else target_ref_feat.clone()
                prev_target_ref_weight = args.ref_weight if (target_ref_feat is not None or target_ref_mse_image is not None) else 0.0
                prev_target_ref_mse_image = target_ref_mse_image
                blend_start_step = step

            current_phase_idx = phase_idx
            phase_start_step = step
            sorted_switches = sorted(switch_steps.keys())
            idx = sorted_switches.index(step)
            if idx + 1 < len(sorted_switches):
                current_phase_len = sorted_switches[idx + 1] - step
            else:
                current_phase_len = args.num_iter - step

            target_text_feat = get_text_feat(phase_idx)
            target_ref_feat = get_ref_feat(phase_idx)
            target_ref_mse_image = get_ref_mse_image(phase_idx)

            label_parts = []
            if prompts:
                label_parts.append(f"prompt={prompts[phase_idx % n_text]!r}")
            if ref_images:
                label_parts.append(f"ref={ref_images[phase_idx % n_ref]}")
            print(f"step={step} switching to phase {phase_idx}: {', '.join(label_parts)}")

            if args.phase_optim_reset:
                optim.state.clear()
                print(f"  optimizer state reset")

            if args.phase_perturb > 0:
                for path in shapes:
                    path.points.data += torch.randn_like(path.points.data) * args.phase_perturb
                print(f"  perturbed points (std={args.phase_perturb})")

            if args.phase_lr_reset:
                for g, blr in zip(optim.param_groups, base_lrs):
                    g["lr"] = blr
                print(f"  LR reset to base")

        step_in_phase = step - phase_start_step
        ref_weight = get_phase_ref_weight(step_in_phase, current_phase_len)

        # Compute effective target (with optional blending)
        if (args.phase_blend > 0 and prev_target_text_feat is not None and target_text_feat is not None
                and step < blend_start_step + args.phase_blend):
            blend_alpha = (step - blend_start_step) / args.phase_blend
            eff_text_feat = (1 - blend_alpha) * prev_target_text_feat + blend_alpha * target_text_feat
        else:
            eff_text_feat = target_text_feat

        if (args.phase_blend > 0 and prev_target_ref_feat is not None and target_ref_feat is not None
                and step < blend_start_step + args.phase_blend):
            blend_alpha = (step - blend_start_step) / args.phase_blend
            eff_ref_feat = (1 - blend_alpha) * prev_target_ref_feat + blend_alpha * target_ref_feat
            eff_ref_weight = (1 - blend_alpha) * prev_target_ref_weight + blend_alpha * ref_weight
        else:
            eff_ref_feat = target_ref_feat
            eff_ref_weight = ref_weight

        if (args.phase_blend > 0 and prev_target_ref_mse_image is not None and target_ref_mse_image is not None
                and step < blend_start_step + args.phase_blend):
            blend_alpha = (step - blend_start_step) / args.phase_blend
            eff_ref_mse_image = (1 - blend_alpha) * prev_target_ref_mse_image + blend_alpha * target_ref_mse_image
            eff_ref_weight = (1 - blend_alpha) * prev_target_ref_weight + blend_alpha * ref_weight
        else:
            eff_ref_mse_image = target_ref_mse_image

        optim.zero_grad(set_to_none=True)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = render(canvas_width, canvas_height, 2, 2, step, None, *scene_args)
        img_alpha = img[:, :, 3:4]
        img = img_alpha * img[:, :, :3] + background * (1 - img_alpha)
        final_img = img

        loss = torch.tensor(0.0, device=device)

        if eff_text_feat is not None:
            prompt_img = apply_gradient_mask(img, prompt_mask)
            clip_img = prompt_img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
            im_batch = torch.cat([augment(clip_img) for _ in range(args.num_augs)])
            with torch.amp.autocast("cuda", dtype=torch.float16):
                image_features = model.encode_image(im_batch)
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                tgt_norm = eff_text_feat / eff_text_feat.norm(dim=-1, keepdim=True)
                sims = im_norm @ tgt_norm.T
                loss = loss - sims.sum()

        if neg_features_list:
            neg_img = apply_gradient_mask(img, prompt_mask)
            clip_img = neg_img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
            im_batch = torch.cat([augment(clip_img) for _ in range(args.num_augs)])
            with torch.amp.autocast("cuda", dtype=torch.float16):
                image_features = model.encode_image(im_batch)
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                for nf in neg_features_list:
                    tgt_norm = nf / nf.norm(dim=-1, keepdim=True)
                    sims = im_norm @ tgt_norm.T
                    loss = loss + sims.sum()

        if eff_ref_feat is not None:
            ref_img = apply_gradient_mask(img, ref_mask)
            clip_img = ref_img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
            im_batch = torch.cat([augment(clip_img) for _ in range(args.num_augs)])

            with torch.amp.autocast("cuda", dtype=torch.float16):
                image_features = model.encode_image(im_batch)
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                tgt_norm = eff_ref_feat / eff_ref_feat.norm(dim=-1, keepdim=True)
                sims = im_norm @ tgt_norm.T
                loss = loss - eff_ref_weight * sims.sum()

        if eff_ref_mse_image is not None:
            phase_ref_mask = None if eff_text_feat is None else ref_mask
            loss = loss + eff_ref_weight * masked_mse_loss(img, eff_ref_mse_image, phase_ref_mask)

        loss.backward()
        optim.step()

        for path in shapes:
            path.stroke_width.data.clamp_(1.0, args.max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if step % args.save_every == 0 or step == args.num_iter - 1:
            png_path = iters_dir / f"iter_{step:04d}.png"
            svg_path = iters_dir / f"iter_{step:04d}.svg"
            latest_png = output_dir / "latest.png"
            latest_svg = output_dir / "latest.svg"
            pydiffvg.imwrite(final_img.detach().cpu(), str(png_path), gamma=1.0)
            pydiffvg.imwrite(final_img.detach().cpu(), str(latest_png), gamma=1.0)
            pydiffvg.save_svg(str(svg_path), canvas_width, canvas_height, shapes, shape_groups)
            pydiffvg.save_svg(str(latest_svg), canvas_width, canvas_height, shapes, shape_groups)
            progress_frames.append(tensor_to_uint8(final_img))
            print(f"step={step} loss={loss.item():.4f} saved = {png_path}")

    final_png = output_dir / "final.png"
    final_svg = output_dir / "final.svg"
    progress_gif = output_dir / "progress.gif"
    progress_mp4 = output_dir / "progress.mp4"
    strokes_gif = output_dir / "strokes.gif"
    strokes_mp4 = output_dir / "strokes.mp4"
    pydiffvg.imwrite(final_img.detach().cpu(), str(final_png), gamma=1.0)
    pydiffvg.save_svg(str(final_svg), canvas_width, canvas_height, shapes, shape_groups)
    write_gif(progress_frames, progress_gif, fps=8)
    write_mp4(progress_frames, progress_mp4, fps=8)

    stroke_frames = []
    for idx in range(len(shapes)):
        stroke_img = render_partial_scene(
            render,
            canvas_width,
            canvas_height,
            shapes[: idx + 1],
            shape_groups[: idx + 1],
            idx,
            background,
        )
        stroke_frames.append(tensor_to_uint8(stroke_img))
    write_gif(stroke_frames, strokes_gif, fps=24)
    write_mp4(stroke_frames, strokes_mp4, fps=24)

    print(f"final_png = {final_png}")
    print(f"final_svg = {final_svg}")
    print(f"progress_gif = {progress_gif}")
    print(f"progress_mp4 = {progress_mp4}")
    print(f"strokes_gif = {strokes_gif}")
    print(f"strokes_mp4 = {strokes_mp4}")


if __name__ == "__main__":
    main()
