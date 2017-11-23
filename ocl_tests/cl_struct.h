#ifndef CL_STRUCT
#define CL_STRUCT

#define NEIGHBOUR_COUNT 32

struct extendet_particle {
  size_t p_id;
  int neigbour_list[NEIGHBOUR_COUNT];
};

#endif