use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResolveError {
    #[error("Symbol is unbound")]
    Unbound,
    #[error("Symbol is already bound")]
    AlreadyBound,
}

#[derive(Debug, Error)]
pub enum ScopedMapError {
    #[error("Key already exists in the current scope")]
    KeyAlreadyExists,
    #[error("Key not found in any scope")]
    KeyNotFound,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TypecheckError {
    #[error("Unexpected type")]
    UnexpectedType,
    #[error("No common supertype found")]
    NoJoin,
    #[error("Invalid binary operation")]
    BinopError,
    #[error("Expression should be in let binder")]
    NotInBinder,
    #[error("Argument length mismatch")]
    ArgLengthMismatch,
    #[error("Incorrect number of dimensions for array access")]
    IncorrectAccessDims,
    #[error("Invalid shrink width")]
    InvalidShrinkWidth,
    #[error("Invalid align factor")]
    InvalidAlignFactor,
    #[error("Pipeline error")]
    PipelineError,
    #[error("Missing field in struct literal")]
    MissingField,
    #[error("Extra fields in struct literal")]
    ExtraFields,
    #[error("Invalid split factor")]
    InvalidSplitFactor,
    #[error("Type is already bound")]
    AlreadyBound,
    #[error("Explicit type annotation is required")]
    ExplicitTypeMissing,
    #[error("Unsupported feature: {0}")]
    Unsupported(&'static str),
    #[error("Array literal length mismatch")]
    LiteralLengthMismatch,
    #[error("Unknown type alias")]
    UnknownAlias,
    #[error("Invalid array dimensions")]
    InvalidArrayDims,
    #[error("Unbound variable")]
    Unbound,
    #[error("Unknown record field")]
    UnknownRecordField,
}

#[derive(Debug, Error)]
pub enum TypeEnvError {
    #[error("Type is unbound")]
    Unbound,
    #[error("Type is already bound")]
    AlreadyBound,
    #[error("Unknown alias")]
    UnknownAlias,
}
