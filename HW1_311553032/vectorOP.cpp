#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_float vw = _pp_vset_float(VECTOR_WIDTH);
  float n = N-1;
  __pp_vec_float  target = _pp_vset_float(n);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;
  float nums[VECTOR_WIDTH];
  for (int i = 0; i < VECTOR_WIDTH; i++){
    float j = i;
    nums[i] = j;
  }
  __pp_vec_float  nums2;
  maskAll = _pp_init_ones();
  _pp_vload_float(nums2, nums, maskAll);
    
  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for ( int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vgt_float(maskAll, nums2, target, maskAll);
    maskAll = _pp_mask_not(maskAll);

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
    _pp_vadd_float(nums2, vw, nums2, maskAll  );

  }

}

void clampedExpVector(float *values, int *exponents, float *output, int N)  {
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //
    
    __pp_vec_float x = _pp_vset_float(0.0);
    __pp_vec_int exp = _pp_vset_int(0);
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_vec_int   zero1 = _pp_vset_int(0);
    __pp_vec_float nine = _pp_vset_float(9.999999f);
    __pp_vec_int one2 = _pp_vset_int(1);
    __pp_vec_int  target = _pp_vset_int(N-1);
    __pp_mask maskAll, maskIsZero, maskIsNotZero, maskIsNien, maskIsNotNien;
    __pp_vec_int vw = _pp_vset_int(VECTOR_WIDTH);
    int nums[VECTOR_WIDTH];
    for (int i = 0; i < VECTOR_WIDTH; i++){
      nums[i] = i;
    }
    __pp_vec_int  nums2;
    maskAll = _pp_init_ones();
    _pp_vload_int(nums2, nums, maskAll);
    
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // All ones
        result = _pp_vset_float(1.f);
        _pp_vgt_int(maskAll, nums2, target, maskAll);
        maskAll = _pp_mask_not(maskAll);

        // All zeros
        maskIsZero = _pp_init_ones(1);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];
        _pp_vload_int(exp, exponents + i, maskAll); // exp = exponents[i]
        _pp_veq_int(maskIsZero, exp, zero1, maskAll); // exp == 0
        maskIsNotZero = _pp_mask_not(maskIsZero); // exp != 0
        maskIsNotZero = _pp_mask_and(maskIsNotZero, maskAll);
        int nZero = _pp_cntbits(maskIsNotZero); // compute exp num
        while( nZero > 0 ) { // expnum > 0
            _pp_vmult_float(result, result, x, maskIsNotZero); // result*=x
            _pp_vsub_int(exp, exp, one2, maskIsNotZero); // exp--
            _pp_veq_int(maskIsZero, exp, zero1, maskIsNotZero);
            maskIsNotZero = _pp_mask_not(maskIsZero); // exp != 0
            maskIsNotZero = _pp_mask_and(maskIsNotZero, maskAll);
            nZero = _pp_cntbits(maskIsNotZero);
        } // while
        
        //result > 9.99999f
        maskIsNien = _pp_init_ones(1);
        _pp_vgt_float(maskIsNien, result, nine, maskAll);
        _pp_vadd_float(result, zero, nine, maskIsNien);
        
        _pp_vstore_float(output + i, result, maskAll);
        _pp_vadd_int(nums2, vw, nums2, maskAll  );
    } // for


} //clampedExpVector()

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //
    __pp_vec_float x = _pp_vset_float(0.0);
    __pp_vec_float result = _pp_vset_float(0.0);
    __pp_vec_int vw = _pp_vset_int(VECTOR_WIDTH);
    __pp_vec_int  target = _pp_vset_int(N-1);
    __pp_mask maskAll;
    int nums[VECTOR_WIDTH];
    for (int i = 0; i < VECTOR_WIDTH; i++){
      nums[i] = i;
    }
    __pp_vec_int  nums2;
    maskAll = _pp_init_ones();
    _pp_vload_int(nums2, nums, maskAll);
    
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        _pp_vload_float(x, values + i, maskAll); // x = values[i];
        _pp_vadd_float(result, result, x, maskAll  ); // result[i]+=x
        _pp_vadd_int(nums2, vw, nums2, maskAll  );
    } // for
    
    float n[VECTOR_WIDTH] = {0.0};
    _pp_vstore_float(n, result, maskAll);
    float sum = 0.0;
    for (int i = 0; i < VECTOR_WIDTH; i++)
        sum += n[i];
    return sum;
}
