#![allow(non_snake_case)]
use super::*;
use cudasys::types::cudnn::*;

gen_hijack!(
    1502,
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

gen_hijack!(
    1504,
    "cudnnSetActivationDescriptor", 
    "cudnnStatus_t", 
    "cudnnActivationDescriptor_t", 
    "cudnnActivationMode_t", 
    "cudnnNanPropagation_t", 
    "f64"
);

gen_hijack!(
    1506,
    "cudnnDestroy", 
    "cudnnStatus_t", 
    "cudnnHandle_t"
);

gen_hijack!(
    1507,
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
gen_hijack_async!(
   1508,
   "cudnnSetStream", 
   "cudnnStatus_t", 
   "cudnnHandle_t", 
   "cudaStream_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
   1508,
   "cudnnSetStream", 
   "cudnnStatus_t", 
   "cudnnHandle_t", 
   "cudaStream_t"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    1510,
    "cudnnDestroyTensorDescriptor", 
    "cudnnStatus_t", 
    "cudnnTensorDescriptor_t"   
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    1510,
    "cudnnDestroyTensorDescriptor", 
    "cudnnStatus_t", 
    "cudnnTensorDescriptor_t"   
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    1512,
    "cudnnDestroyFilterDescriptor",
    "cudnnStatus_t",
    "cudnnFilterDescriptor_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    1512,
    "cudnnDestroyFilterDescriptor",
    "cudnnStatus_t",
    "cudnnFilterDescriptor_t"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    1515,
    "cudnnDestroyConvolutionDescriptor",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    1515,
    "cudnnDestroyConvolutionDescriptor",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    1517,
    "cudnnSetConvolutionGroupCount",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t",
    "::std::os::raw::c_int"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    1517,
    "cudnnSetConvolutionGroupCount",
    "cudnnStatus_t",
    "cudnnConvolutionDescriptor_t",
    "::std::os::raw::c_int"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    1518,
    "cudnnSetConvolutionMathType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnMathType_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    1518,
    "cudnnSetConvolutionMathType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnMathType_t"
);

gen_hijack!(
    1519,
    "cudnnSetConvolutionReorderType", 
    "cudnnStatus_t", 
    "cudnnConvolutionDescriptor_t", 
    "cudnnReorderType_t"
);