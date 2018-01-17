#ifdef cl_amd_printf
	#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PRINTF_ON // this comment because using printf leads to very slow work on Radeon r9 290x on my machine
                  // don't know why
#elif defined(cl_intel_printf)
	#pragma OPENCL EXTENSION cl_intel_printf : enable
#define PRINTF_ON
#endif

#ifdef _DOUBLE_PRECISION
	#ifdef cl_khr_fp64
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		
	#elif defined(cl_amd_fp64)
		#pragma OPENCL EXTENSION cl_amd_fp64 : enable
		#define _DOUBLE_PRECISION
	#else
		#error "Double precision floating point not supported by OpenCL implementation."
	#endif
#endif

#include "cl_struct.h"

typedef struct particle_f{
	float4 pos;
	float4 vel;
	size_t cell_id;
	size_t type_;
	float density;
	float pressure;
} particle_f;

typedef struct particle_d{
	double4 pos;
	double4 vel;
	size_t cell_id;
	size_t type_;
	double density;
	double pressure;
} particle_d;


__kernel void work_with_struct(__global struct extendet_particle * ext_particles, 
							   __global struct 
							   #ifdef _DOUBLE_PRECISION
									particle_d
							   #else
									particle_f
							   #endif 
									* particles){
	int id = get_global_id(0);
#ifdef PRINTF_ON
	if(id == 0){
		printf("sizeof() of particles_f is %d\n", sizeof(particle_f) );
		printf("sizeof() of particles_d is %d\n", sizeof(particle_d) );
	}
#endif
#ifdef _DOUBLE_PRECISION

	particles[id].pos = (double4)(id, id, id, id);
	particles[id].vel = (double4)(id, id, id, id);
#else
	particles[id].pos = (float4)(id, id, id, id);
	particles[id].vel = (float4)(id, id, id, id);
#endif
	particles[id].type_ = id + 1;
}


__kernel void _init_ext_particles(__global struct extendet_particle * ext_particles){
	int id = get_global_id(0);
	ext_particles[id].p_id = id;
	for(int i=0;i<NEIGHBOUR_COUNT;++i){
		ext_particles[id].neigbour_list[i] = -1;
	}
}