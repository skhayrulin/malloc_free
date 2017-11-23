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

__kernel void ker_blur(__global float4* img, __global float4 * ret_img, uint cols, uint rows){
	int id = get_global_id(0);
	int jd = get_global_id(1);
	//int id = get_global_id(0);
	//float sum = 0.0f;
    /*for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
		int4 col = read_imagei(img, sampler, pos);
            sum += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]
                *read_imagef(image, sampler, pos + (int2)(a,b)).x;
        }
    /*}
	
	ret_img[pos.x+pos.y*get_global_size(1)] = sum;*/
	if(get_global_size(1) != 0){
		//printf("pos %d, %d\n",id, jd);
		//printf("pos %d, %d\n",get_global_size(0), get_global_size(1));
	//	printf("size %f %f %f %f  \n",img[id].x,   img[id].y , img[id].z, img[id].w);
	}
	//float4 color = read_imagef(img, sampler, pos);
	//ret_img[id]= img[id];
	//ret_img[id*get_global_size(0) + jd]= (float4)(1.0f,1.0f,0.0f,0.0f);
	ret_img[id *cols  + jd]= img[id*cols + jd];
	//printf("size %f %f %f %f  \n",ret_img[id].x,   ret_img[id].y , ret_img[id].z, ret_img[id].w);
	//ret_img[pos.x + pos.y*get_global_size(1)]= (float4)(1.0f,0.0f,0.0f,0.0f);//color.x;
	//ret_img[pos.x] + pos.y*get_global_size(1) + 1]= 1.0f;//color.x;
	//ret_img[pos.x] + pos.y*get_global_size(1) + 2]= 1.0f;//color.x;
}