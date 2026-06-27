from  pathlib import Path
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

#folder_name = "room_from_webgl_3dgs_vercel_app"
#folder_name = "original"
#folder_name = "kitti_10img_da3_render"
# folder_name = "srikanth_lab_gsplat_render"
folder_name = "mipnerf_bicycle_extreme_pose_variation"

input_path = Path(f"examples/single_img_data/{folder_name}/input.png")
ref_path = Path(f"examples/single_img_data/{folder_name}/ref.png")

prompt = "remove degradation"

def main():

    if not input_path.exists():
        print(f"input image not found at path: {input_path}")
        return
    input_image = load_image(str(input_path))
    print(f"Loaded input image from path: {input_path}")

    # difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    # difix.to("cuda")
    # print(f"\n\nLoaded DifixPipeline for Difix (without ref image)")

    # for timestep in [49,99,149,199,299,399,499,599,799,999]:
    # # for timestep in [199]:
    
    #     print (f"Running Difix without ref image for timestep: {timestep}")
    #     output_image = difix(prompt, image=input_image, num_inference_steps=1, timesteps=[timestep], guidance_scale=0.0).images[0]
    #     output_path = Path(f"examples/single_img_results/{folder_name}/without_ref/output_timestep_{timestep}.png")
    #     output_path.parent.mkdir(parents=True, exist_ok=True)   
    #     output_image.save(str(output_path))


    if not ref_path.exists():
        print(f"ref image not found at path: {ref_path}")
        return
    ref_image = load_image(str(ref_path))
    print(f"Loaded ref image from path: {ref_path}")

    difix_ref = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    difix_ref.to("cuda")
    print(f"\n\nLoaded DifixPipeline for Difix (with ref image)")
    

    # for timestep in [49,99,149,199,299,399,499,599,799,999]:
    for timestep in [999]:
        print (f"Running Difix with ref_image for timestep: {timestep}")
        output_using_ref_image = difix_ref(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[timestep], guidance_scale=0.0).images[0]
        output_using_ref_path = Path(f"examples/single_img_results/{folder_name}/with_ref/output_timestep_{timestep}.png")
        output_using_ref_path.parent.mkdir(parents=True, exist_ok=True)
        output_using_ref_image.save(str(output_using_ref_path))
    
if __name__ == "__main__":
    main()
