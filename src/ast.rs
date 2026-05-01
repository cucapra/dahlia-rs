use std::cell::RefCell;
use std::collections::HashMap;

use cranelift_entity::{EntityList, ListPool, PrimaryMap, entity_impl};

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ExprId(u32);
entity_impl!(ExprId, "expr");

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct CommandId(u32);
entity_impl!(CommandId, "cmd");

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct TypeId(u32);
entity_impl!(TypeId, "typ");

#[derive(Debug, Default)]
pub struct Context {
    pub exprs: PrimaryMap<ExprId, Expr>,
    pub commands: PrimaryMap<CommandId, Command>,
    pub expr_lists: ListPool<ExprId>,
    pub command_lists: ListPool<CommandId>,
    pub types: PrimaryMap<TypeId, Type>,
    pub type_lists: ListPool<TypeId>,
    pub type_map: HashMap<TypeKey, TypeId>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Id(pub String);

#[derive(Debug, Clone)]
pub enum Type {
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
    Array {
        element_type: TypeId,
        dims: Vec<DimSpec>,
        ports: usize,
    },
    StaticInt(u64), // the original Scala repo only seems to use this for unsigned bit types
    Index {
        static_: (i64, i64),
        dynamic: (i64, i64),
    },
    Void,
    Rational(String),
    Func {
        args: EntityList<TypeId>,
        ret: TypeId,
    },
    RecType {
        name: Id,
        fields: HashMap<Id, TypeId>,
    },
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum TypeKey {
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
    Array {
        element_type: TypeId,
        dims: Vec<DimSpec>,
        ports: usize,
    },
    StaticInt(u64),
    Index {
        static_: (i64, i64),
        dynamic: (i64, i64),
    },
    Void,
    Rational(String),
    Func {
        args: Vec<TypeId>,
        ret: TypeId,
    },
    RecType {
        name: Id,
        fields: Vec<(Id, TypeId)>,
    },
}

impl Context {
    pub fn get_float(&mut self) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Float)
            .or_insert_with(|| self.types.push(Type::Float))
            .clone()
    }

    pub fn get_double(&mut self) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Double)
            .or_insert_with(|| self.types.push(Type::Double))
            .clone()
    }

    pub fn get_bool(&mut self) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Bool)
            .or_insert_with(|| self.types.push(Type::Bool))
            .clone()
    }

    pub fn get_bit(&mut self, length: usize, unsigned: bool) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Bit { length, unsigned })
            .or_insert_with(|| self.types.push(Type::Bit { length, unsigned }))
            .clone()
    }

    pub fn get_fixed(
        &mut self,
        length_total: usize,
        length_int: usize,
        unsigned: bool,
    ) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Fixed {
                length_total,
                length_int,
                unsigned,
            })
            .or_insert_with(|| {
                self.types.push(Type::Fixed {
                    length_total,
                    length_int,
                    unsigned,
                })
            })
            .clone()
    }

    pub fn get_alias(&mut self, name: Id) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Alias(name.clone()))
            .or_insert_with(|| self.types.push(Type::Alias(name)))
            .clone()
    }

    pub fn get_array(&mut self, element_type: TypeId, dims: Vec<DimSpec>, ports: usize) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Array {
                element_type,
                dims: dims.clone(),
                ports,
            })
            .or_insert_with(|| self.types.push(Type::Array { element_type, dims, ports }))
            .clone()
    }

    pub fn get_static_int(&mut self, value: u64) -> TypeId {
        self
            .type_map
            .entry(TypeKey::StaticInt(value))
            .or_insert_with(|| self.types.push(Type::StaticInt(value)))
            .clone()
    }

    pub fn get_index(&mut self, static_: (i64, i64), dynamic: (i64, i64)) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Index { static_, dynamic })
            .or_insert_with(|| self.types.push(Type::Index { static_, dynamic }))
            .clone()
    }

    pub fn get_void(&mut self) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Void)
            .or_insert_with(|| self.types.push(Type::Void))
            .clone()
    }

    pub fn get_rational(&mut self, value: String) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Rational(value.clone()))
            .or_insert_with(|| self.types.push(Type::Rational(value)))
            .clone()
    }

    pub fn get_func(&mut self, args: Vec<TypeId>, ret: TypeId) -> TypeId {
        self
            .type_map
            .entry(TypeKey::Func {
                args: args.clone(),
                ret,
            })
            .or_insert_with(|| {
                self.types.push(Type::Func {
                    args: EntityList::from_iter(args, &mut self.type_lists),
                    ret,
                })
            })
            .clone()
    }

    pub fn get_rec_type(&mut self, name: Id, fields: HashMap<Id, TypeId>) -> TypeId {
        self
            .type_map
            .entry(TypeKey::RecType {
                name: name.clone(),
                fields: fields.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            })
            .or_insert_with(|| self.types.push(Type::RecType { name, fields }))
            .clone()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DimSpec {
    pub length: usize,
    pub bank: Option<usize>,
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
        expr: ExprId,
        ty: TypeId,
    },

    ArrayLiteral(EntityList<ExprId>),
    RecordLiteral(HashMap<Id, ExprId>),

    RationalLiteral(String),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: Id,
        indices: EntityList<ExprId>,
    },
    RecordAccess {
        record: ExprId,
        field: Id,
    },

    Application {
        func: Id,
        args: EntityList<ExprId>,
    },

    Id(Id),

    BinOp {
        left: ExprId,
        op: InfixOp,
        right: ExprId,
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
    pub ty: Option<TypeId>,
    pub rev: bool,
    pub start: usize,
    pub end: usize,
    pub unroll: usize,
}

#[derive(Debug)]
pub enum Command {
    Empty,
    Par(Vec<CommandId>),
    Seq(Vec<CommandId>),
    Let {
        id: Id,
        ty: Option<TypeId>,
        value: Option<ExprId>,
    },
    Update {
        lhs: ExprId,
        op: AssignOp,
        rhs: ExprId,
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
    Return(ExprId),
    IfElse {
        cond: ExprId,
        then: CommandId,
        else_: CommandId,
    },
    While {
        cond: ExprId,
        pipeline: bool,
        body: CommandId,
    },
    For {
        range: ForRange,
        pipeline: bool,
        body: CommandId,
        combine: CommandId,
    },
    Decorate(String),
    Expr(ExprId),
}

#[derive(Debug)]
pub struct Decl {
    pub id: Id,
    pub ty: TypeId,
}

#[derive(Debug)]
pub struct FuncSig {
    pub name: Id,
    pub args: Vec<Decl>,
    pub ret_ty: Option<TypeId>,
}

#[derive(Debug)]
pub enum Def {
    Func { sig: FuncSig, body: CommandId },
    Record { name: Id, fields: Vec<Decl> },
}

#[derive(Debug)]
pub enum Suffix {
    Rotation(ExprId),
    Aligned { factor: usize, e: ExprId },
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
    pub decors: EntityList<CommandId>,
    pub decls: Vec<Decl>,
    pub cmd: CommandId,
}

impl Command {
    pub fn smart_par(cmds: Vec<CommandId>, context: &RefCell<Context>) -> CommandId {
        let mut flat = Vec::new();
        {
            let context = context.borrow();
            for cmd in cmds {
                match &context.commands[cmd] {
                    Command::Par(cs) => flat.extend(cs),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            context.borrow_mut().commands.push(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut context = context.borrow_mut();
            context.commands.push(Command::Par(flat))
        }
    }

    pub fn smart_seq(cmds: Vec<CommandId>, context: &RefCell<Context>) -> CommandId {
        let mut flat = Vec::new();
        {
            let context = context.borrow();
            for cmd in cmds {
                match &context.commands[cmd] {
                    Command::Seq(cs) => flat.extend(cs),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            context.borrow_mut().commands.push(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut context = context.borrow_mut();
            context.commands.push(Command::Seq(flat))
        }
    }
}
