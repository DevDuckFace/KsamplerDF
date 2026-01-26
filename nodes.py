from __future__ import annotations
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

class KsamplerDF(ComfyNodeABC):
    """
    Custom KSampler with Refiner functionality for ComfyUI.
    Splits steps between main sampler and refiner sampler.
    Example: 12 total steps with refiner at step 6 = 6 main + 6 refiner steps.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "The model used for denoising the input latent."}),
                "positive": (IO.CONDITIONING, {"tooltip": "The conditioning describing the attributes to include."}),
                "negative": (IO.CONDITIONING, {"tooltip": "The conditioning describing the attributes to exclude."}),
                "latent_image": (IO.LATENT, {"tooltip": "The latent image to denoise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Total number of steps (main + refiner)."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The algorithm used for the main sampling phase."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "The scheduler for the main sampling phase."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Main denoise amount."}),
                "refiner_at_step": ("INT", {"default": 10, "min": 0, "max": 10000, "tooltip": "Step at which to switch to refiner sampler."}),
                "sampler_refiner": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The algorithm used for the refiner phase."}),
                "scheduler_refiner": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "The scheduler for the refiner phase."}),
                "denoise_refiner": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Refiner denoise - controls how much the refiner interferes (0.0 = no effect, 1.0 = full effect)."}),
            },
        }

    RETURN_TYPES = (IO.LATENT,)
    OUTPUT_TOOLTIPS = ("The denoised latent image.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "KSampler with Refiner - Splits steps between main sampler and refiner sampler with independent denoise control."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, 
               sampler_name, scheduler, denoise, refiner_at_step, sampler_refiner, 
               scheduler_refiner, denoise_refiner):
        
        import nodes
        
        # Clamp refiner_at_step to valid range
        refiner_at_step = max(0, min(refiner_at_step, steps))
        
        # Calculate steps for each phase
        main_steps = refiner_at_step
        refiner_steps = steps - refiner_at_step
        
        current_latent = latent_image
        
        # Phase 1: Main sampling (if main_steps > 0)
        if main_steps > 0:
            print(f"[KSampler DF] Main phase: {main_steps} steps with {sampler_name}/{scheduler}, denoise={denoise}")
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

        # Phase 2: Refiner sampling (if refiner_steps > 0 and denoise_refiner > 0)
        if refiner_steps > 0 and denoise_refiner > 0:
            print(f"[KSampler DF] Refiner phase: {refiner_steps} steps with {sampler_refiner}/{scheduler_refiner}, denoise={denoise_refiner}")
            result = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=refiner_steps,
                cfg=cfg,
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
        elif main_steps > 0:
            # If no refiner, we need to do a final denoise on main result
            # Run one more step to finalize
            print(f"[KSampler DF] Finalizing without refiner...")
            result = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=1,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=current_latent,
                denoise=0.0,
                disable_noise=True,
                start_step=None,
                last_step=None,
                force_full_denoise=True
            )
            current_latent = result[0]

        print(f"[KSampler DF] Complete! Total: {steps} steps ({main_steps} main + {refiner_steps} refiner)")
        
        return (current_latent,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "KsamplerDF": KsamplerDF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KsamplerDF": "KSampler DF (with Refiner)",
}
