use std::collections::HashMap;

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct Id(pub String);

#[derive(Debug)]
pub enum TypeAtom {
    Float,
    Double,
    Bool,
    Bit {
        length: usize,
        unsigned: bool,
    },
    Fixed {
        length_total: usize,
        length_int: usize,
        unsigned: bool,
    },
    Alias(Id),
}

#[derive(Debug)]
pub struct DimSpec {
    pub length: usize,
    pub bank: Option<usize>,
}

#[derive(Debug)]
pub enum Type {
    Simple(TypeAtom),
    Array {
        element_type: TypeAtom,
        dims: Vec<DimSpec>,
        ports: usize,
    },
}

#[derive(Debug)]
pub enum InfixOp {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Shl,
    Shr,
    Eq,
    Neq,
    Le,
    Ge,
    Lt,
    Gt,
    And,
    Or,
    Band,
    Bor,
    Bxor,
}

#[derive(Debug)]
pub enum Expr {
    Cast {
        expr: Box<Expr>,
        ty: TypeAtom,
    },

    ArrayLiteral(Vec<Expr>),
    RecordLiteral(HashMap<Id, Expr>),

    RationalLiteral(String),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: Id,
        indices: Vec<Expr>,
    },
    RecordAccess {
        record: Box<Expr>,
        field: Id,
    },

    Application {
        func: Id,
        args: Vec<Expr>,
    },

    Id(Id),

    BinOp {
        left: Box<Expr>,
        op: InfixOp,
        right: Box<Expr>,
    },
}

#[derive(Debug)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

#[derive(Debug)]
pub struct ForRange {
    pub id: Id,
    pub ty: Option<Type>,
    pub rev: bool,
    pub start: usize,
    pub end: usize,
    pub unroll: usize,
}

#[derive(Debug)]
pub enum Command {
    Par(Vec<Command>),
    Seq(Vec<Command>),
    Let {
        id: Id,
        ty: Option<Type>,
        value: Option<Expr>,
    },
    Update {
        lhs: Expr,
        op: AssignOp,
        rhs: Expr,
    },
    View {
        id: Id,
        arr_id: Id,
        dims: Vec<View>,
    },
    Split {
        id: Id,
        arr_id: Id,
        dims: Vec<usize>,
    },
    Return(Expr),
    IfElse {
        cond: Expr,
        then: Box<Command>,
        else_: Option<Box<Command>>,
    },
    While {
        cond: Expr,
        pipeline: bool,
        body: Box<Command>,
    },
    For {
        range: ForRange,
        pipeline: bool,
        body: Box<Command>,
        combine: Option<Box<Command>>,
    },
    Decorate(String),
    Expr(Expr),
}

#[derive(Debug)]
pub struct Decl {
    pub id: Id,
    pub ty: Type,
}

#[derive(Debug)]
pub struct FuncSig {
    pub name: Id,
    pub args: Vec<Decl>,
    pub ret_ty: Option<Type>,
}

#[derive(Debug)]
pub enum Def {
    Func { sig: FuncSig, body: Command },
    Record { name: Id, fields: Vec<Decl> },
}

#[derive(Debug)]
pub enum Suffix {
    Rotation(Expr),
    Aligned { factor: usize, e: Expr },
}

#[derive(Debug)]
pub struct View {
    pub suffix: Suffix,
    pub prefix: Option<usize>,
    pub shrink: Option<usize>,
}

#[derive(Debug)]
pub enum Backend {
    Cpp,
    Vivado,
    Futil,
    Calyx,
}

#[derive(Debug)]
pub struct Include {
    pub backends: Vec<(Backend, String)>,
    pub defs: Vec<FuncSig>,
}

#[derive(Debug)]
pub struct Program {
    pub includes: Vec<Include>,
    pub defs: Vec<Def>,
    pub decors: Vec<Command>,
    pub decls: Vec<Decl>,
    pub cmd: Option<Command>,
}
