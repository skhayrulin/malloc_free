#ifdef cl_amd_printf
	#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PRINTF_ON // this comment because using printf leads to very slow work on Radeon r9 290x on my machine
                  // don't know why
#elif defined(cl_intel_printf)
	#pragma OPENCL EXTENSION cl_intel_printf : enable
#define PRINTF_ON
#endif
//__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
//__read_only image2d_t img

__kernel void ker_blur(__global float4* img, __global float4 * ret_img, __global float* mask, uint cols, uint rows){
	int id = get_global_id(0);
	if(id == 0 || id == get_global_size(0))
		return;
	int jd = get_global_id(1);
	if(jd == 0 || jd == get_global_size(1))
		return;
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for(int a = -1; a <= 1; a++) {
        for(int b = -1; b <= 1; b++) {
			float k = mask[(a + 1) * 3 + b + 1];
			int ij = (id * get_global_size(0)+jd) + get_global_size(0) * a + b;
			sum += k * img[ij];
        }
    }
	
	ret_img[id * get_global_size(0)+jd] = sum;
	//ret_img[id *cols  + jd]= img[id*cols + jd];
	
}