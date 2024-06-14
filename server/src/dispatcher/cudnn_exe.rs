#![allow(non_snake_case)]
use super::*;
use cudasys::cudnn::*;

gen_exe!(
    "cudnnSetTensor4dDescriptor", 
    "cudnnStatus_t", 
    "cudnnTensorDescriptor_t", 
    "cudnnTensorFormat_t", 
    "cudnnDataType_t", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int"
);

gen_exe!(
    "cudnnSetActivationDescriptor", 
    "cudnnStatus_t", 
    "cudnnActivationDescriptor_t", 
    "cudnnActivationMode_t", 
    "cudnnNanPropagation_t", 
    "f64"
);

gen_exe!(
    "cudnnDestroy", 
    "cudnnStatus_t", 
    "cudnnHandle_t"
);

gen_exe!(
    "cudnnSetConvolution2dDescriptor", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "::std::os::raw::c_int", 
    "cudnnConvolutionMode_t", 
    "cudnnDataType_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
   "cudnnSetStream", 
   "cudnnStatus_t", 
   "cudnnHandle_t", 
   "cudaStream_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
   "cudnnSetStream", 
   "cudnnStatus_t", 
   "cudnnHandle_t", 
   "cudaStream_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cudnnDestroyTensorDescriptor", 
    "cudnnStatus_t", 
    "cudnnTensorDescriptor_t"   
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cudnnDestroyTensorDescriptor", 
    "cudnnStatus_t", 
    "cudnnTensorDescriptor_t"   
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cudnnDestroyFilterDescriptor",
    "cudnnStatus_t",
    "cudnnFilterDescriptor_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cudnnDestroyFilterDescriptor",
    "cudnnStatus_t",
    "cudnnFilterDescriptor_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cudnnDestroyConvolutionDescriptor",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cudnnDestroyConvolutionDescriptor",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cudnnSetConvolutionGroupCount",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t",
    "::std::os::raw::c_int"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cudnnSetConvolutionGroupCount",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t",
    "::std::os::raw::c_int"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cudnnSetConvolutionMathType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnMathType_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cudnnSetConvolutionMathType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnMathType_t"
);

gen_exe!(
    "cudnnSetConvolutionReorderType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnReorderType_t"
);

gen_exe!(
    "cudnnSetFilter4dDescriptor",
    "cudnnStatus_t",
    "cudnnFilterDescriptor_t",
    "cudnnDataType_t",
    "cudnnTensorFormat_t",
    "::std::os::raw::c_int",
    "::std::os::raw::c_int",
    "::std::os::raw::c_int",
    "::std::os::raw::c_int"
);