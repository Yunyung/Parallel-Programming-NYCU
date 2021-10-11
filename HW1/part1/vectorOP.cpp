#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

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
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_int exp;
  __pp_vec_int expCounter;
  __pp_vec_float clampedMaxThres = _pp_vset_float(9.999999f);
  __pp_vec_float result;
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_mask maskAll, maskExp, maskCounterIsPositive, maskNeedClamped;

  // Deal with remaing element when N % VECTOR_WIDTH != 0
  // int remaining = N % VECTOR_WIDTH;
  for (int i = N;i < N + VECTOR_WIDTH;i++) {
    values[i] = 0.0f;
    exponents[i] = 1;
  }

  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();

    // Init exp mask val to zeros
    maskExp = _pp_init_ones(0);

    expCounter = _pp_vset_int(0);

    _pp_vload_float(x, values + i, maskAll);         // x = values[i:i:VECTOR_WIDTH]

    _pp_vload_int(exp, exponents + i, maskAll);     // exp = exponents[i:i:VECTOR_WIDTH]

    _pp_veq_int(maskExp, exp, zero, maskAll);       // if (exp[j] == 0) {

    _pp_vset_float(result, 1.0f, maskExp);          //   result[j] = 1.0f 

    maskExp = _pp_mask_not(maskExp); // (flip mask) // } else {  

    _pp_vmove_float(result, x, maskExp);            //   result[j] = x[j] }

    _pp_vsub_int(expCounter, exp, one, maskExp);    //   expCounter = exp - 1

    _pp_vgt_int(maskCounterIsPositive, expCounter, zero, maskAll); // maskCounterIsPositive = expCounter > 0

    while (_pp_cntbits(maskCounterIsPositive)) {
      _pp_vmult_float(result, result, x, maskCounterIsPositive);        // result = result * x
      
      _pp_vsub_int(expCounter, expCounter, one, maskCounterIsPositive); // expCounter = expCounter - 1

      _pp_vgt_int(maskCounterIsPositive, expCounter, zero, maskAll);    // maskCounterIsPositive = expCounter > 0
    }
    
    maskNeedClamped = _pp_init_ones(0);

    _pp_vgt_float(maskNeedClamped, result, clampedMaxThres, maskAll); // result = result > clampedMaxThres

    _pp_vset_float(result, 9.999999f, maskNeedClamped); // result[j] = 9.999999f

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll); // output[i:VECTOR_WIDTH] = result
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  
  __pp_vec_float x, result;
  __pp_mask maskAll;

  float ret[VECTOR_WIDTH];
  int shiftCounter = VECTOR_WIDTH;
  maskAll = _pp_init_ones();
  
  // init result to 0
  result = _pp_vset_float(0.0f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll);

    _pp_vadd_float(result, result, x, maskAll);
  }

  // run log(VECTOR_WIDTH) times
  while(shiftCounter != 1) {
    _pp_hadd_float(result, result);
    
    _pp_interleave_float(result, result);

    shiftCounter >>= 1;
  }

  _pp_vstore_float(ret, result, maskAll);

  return ret[0];
}