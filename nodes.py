from __future__ import annotations
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor


class KsamplerDF(ComfyNodeABC):
    """
    Custom KSampler with Refiner and Upscale functionality for ComfyUI.
    Splits steps between main sampler and refiner sampler.
    Includes latent upscale between phases and optional model upscale at the end.
    """
    
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # Get list of available upscale models
        upscale_models_list = ["none"] + folder_paths.get_filename_list("upscale_models")
        
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "The model used for denoising the input latent."}),
                "positive": (IO.CONDITIONING, {"tooltip": "The conditioning describing the attributes to include."}),
                "negative": (IO.CONDITIONING, {"tooltip": "The conditioning describing the attributes to exclude."}),
                "latent_image": (IO.LATENT, {"tooltip": "The latent image to denoise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Total number of steps (main + refiner)."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale for main phase."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The algorithm used for the main sampling phase."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "The scheduler for the main sampling phase."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Main denoise amount."}),
                
                # Latent Upscale between main and refiner
                "latent_upscale_enabled": (["enable", "disable"], {"default": "disable", "tooltip": "Enable latent upscale between main and refiner phases."}),
                "latent_upscale_method": (cls.upscale_methods, {"default": "bislerp", "tooltip": "Upscale method for latent."}),
                "latent_upscale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Scale factor for latent upscale."}),
                
                # Refiner settings
                "refiner_at_step": ("INT", {"default": 10, "min": 0, "max": 10000, "tooltip": "Step at which to switch to refiner sampler."}),
                "sampler_refiner": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The algorithm used for the refiner phase."}),
                "scheduler_refiner": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "The scheduler for the refiner phase."}),
                "cfg_refiner": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale for refiner phase."}),
                "denoise_refiner": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Refiner denoise - controls how much the refiner interferes (0.0 = no effect, 1.0 = full effect)."}),
                
                # VAE for decoding latent to image
                "vae": ("VAE", {"tooltip": "VAE for decoding latent to image output."}),
                
                # Model Upscale at the end - select model directly
                "upscale_model_name": (upscale_models_list, {"default": "none", "tooltip": "Select upscale model (e.g., RealESRGAN, ESRGAN). Select 'none' to disable."}),
            },
        }

    RETURN_TYPES = (IO.LATENT, "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE")
    OUTPUT_TOOLTIPS = ("The denoised latent image.", "The decoded/upscaled image.")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "KSampler with Refiner and Upscale - Splits steps between main and refiner, with latent upscale and optional model upscale."

    def load_upscale_model(self, model_name):
        """Load an upscale model from the upscale_models folder."""
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()
        
        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        
        return out

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, 
               sampler_name, scheduler, denoise, 
               latent_upscale_enabled, latent_upscale_method, latent_upscale_factor,
               refiner_at_step, sampler_refiner, scheduler_refiner, cfg_refiner, denoise_refiner,
               vae, upscale_model_name):
        
        import nodes
        
        # Clamp refiner_at_step to valid range
        refiner_at_step = max(0, min(refiner_at_step, steps))
        
        # Calculate steps for each phase
        main_steps = refiner_at_step
        refiner_steps = steps - refiner_at_step
        
        current_latent = latent_image
        output_image = None
        
        # Phase 1: Main sampling (if main_steps > 0)
        if main_steps > 0:
            print(f"[KSampler DF] Main phase: {main_steps} steps with {sampler_name}/{scheduler}, CFG={cfg}, denoise={denoise}")
            result = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=main_steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=current_latent,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False  # Don't fully denoise, refiner continues
            )
            current_latent = result[0]

        # Phase 2: Latent Upscale (if enabled and before refiner)
        if latent_upscale_enabled == "enable" and refiner_steps > 0:
            print(f"[KSampler DF] Latent upscale: {latent_upscale_method} x{latent_upscale_factor}")
            samples = current_latent["samples"]
            width = round(samples.shape[-1] * latent_upscale_factor)
            height = round(samples.shape[-2] * latent_upscale_factor)
            upscaled = comfy.utils.common_upscale(samples, width, height, latent_upscale_method, "disabled")
            current_latent = current_latent.copy()
            current_latent["samples"] = upscaled

        # Phase 3: Refiner sampling (if refiner_steps > 0 and denoise_refiner > 0)
        if refiner_steps > 0 and denoise_refiner > 0:
            print(f"[KSampler DF] Refiner phase: {refiner_steps} steps with {sampler_refiner}/{scheduler_refiner}, CFG={cfg_refiner}, denoise={denoise_refiner}")
            result = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=refiner_steps,
                cfg=cfg_refiner,
                sampler_name=sampler_refiner,
                scheduler=scheduler_refiner,
                positive=positive,
                negative=negative,
                latent=current_latent,
                denoise=denoise_refiner,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=True  # Final phase, full denoise
            )
            current_latent = result[0]

        print(f"[KSampler DF] Sampling complete! Total: {steps} steps ({main_steps} main + {refiner_steps} refiner)")

        # Decode latent to image using VAE
        samples = current_latent["samples"]
        output_image = vae.decode(samples)
        print(f"[KSampler DF] Decoded latent to image")

        # Phase 4: Model Upscale (if a model is selected)
        if upscale_model_name != "none":
            print(f"[KSampler DF] Loading upscale model: {upscale_model_name}")
            upscale_model = self.load_upscale_model(upscale_model_name)
            
            print(f"[KSampler DF] Applying model upscale...")
            
            # Apply upscale model
            device = comfy.model_management.get_torch_device()
            memory_required = comfy.model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * output_image.element_size() * max(upscale_model.scale, 1.0) * 384.0
            memory_required += output_image.nelement() * output_image.element_size()
            comfy.model_management.free_memory(memory_required, device)
            
            upscale_model.to(device)
            in_img = output_image.movedim(-1, -3).to(device)
            
            tile = 512
            overlap = 32
            oom = True
            
            try:
                while oom:
                    try:
                        steps_upscale = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                            in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                        )
                        pbar = comfy.utils.ProgressBar(steps_upscale)
                        s = comfy.utils.tiled_scale(
                            in_img, lambda a: upscale_model(a), 
                            tile_x=tile, tile_y=tile, overlap=overlap, 
                            upscale_amount=upscale_model.scale, pbar=pbar
                        )
                        oom = False
                    except comfy.model_management.OOM_EXCEPTION as e:
                        tile //= 2
                        if tile < 128:
                            raise e
            finally:
                upscale_model.to("cpu")
            
            output_image = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            print(f"[KSampler DF] Model upscale complete!")
        
        return (current_latent, output_image)


# Register the node
NODE_CLASS_MAPPINGS = {
    "KsamplerDF": KsamplerDF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KsamplerDF": "KSampler DF (with Refiner & Upscale)",
}
