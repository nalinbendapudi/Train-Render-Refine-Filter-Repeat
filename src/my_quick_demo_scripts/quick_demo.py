from  pathlib import Path
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

#folder_name = "room_from_webgl_3dgs_vercel_app"
#folder_name = "original"
#folder_name = "kitti_10img_da3_render"
folder_name = "srikanth_lab_gsplat_render"
input_path = Path(f"examples/single_img_data/{folder_name}/example_input.png")
ref_path = Path(f"examples/single_img_data/{folder_name}/example_ref.png")
output_path = Path(f"examples/single_img_results/{folder_name}/example_output.png")
output_using_ref_path = Path(f"examples/single_img_results/{folder_name}/example_output_using_ref.png")
prompt = "remove degradation"


def main():
    if not input_path.exists():
        print(f"input image not found at path: {input_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_using_ref_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = load_image(str(input_path))

    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save(str(output_path))

    if not ref_path.exists():
        print(f"ref image not found at path: {ref_path}")
        return

    ref_image = load_image(str(ref_path))

    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to("cuda")
    
    print(pipe.__class__)
    print(pipe.__class__.__name__)
    print(pipe.__class__.__module__)

    output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save(str(output_using_ref_path))
    
if __name__ == "__main__":
    main()
