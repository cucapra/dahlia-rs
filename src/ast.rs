use std::collections::HashMap;
use std::{cell::RefCell, hash::Hash};

use cranelift_entity::{EntityList, ListPool, PrimaryMap, entity_impl};
use indexmap::IndexMap;

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct ExprId(u32);
entity_impl!(ExprId, "expr");

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct CommandId(u32);
entity_impl!(CommandId, "command");

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct Symbol(u32);
entity_impl!(Symbol, "symbol");

pub trait IdResolve {
    fn resolve_id<'a>(&self, ast: &'a Ast) -> &'a str;
    fn get_name(&self) -> String;
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct FuncId(u32);
entity_impl!(FuncId, "func");

impl IdResolve for FuncId {
    fn resolve_id<'a>(&self, ast: &'a Ast) -> &'a str {
        ast.symbols
            .get(*ast.funcs.get(*self).expect("Function ID not found"))
            .expect("symbol not found")
    }

    fn get_name(&self) -> String {
        format!("fn_{}", self.as_u32())
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct ValueId(u32);
entity_impl!(ValueId, "value");

impl IdResolve for ValueId {
    fn resolve_id<'a>(&self, ast: &'a Ast) -> &'a str {
        ast.symbols
            .get(*ast.values.get(*self).expect("Value ID not found"))
            .expect("symbol not found")
    }

    fn get_name(&self) -> String {
        format!("v_{}", self.as_u32())
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct RecordId(u32);
entity_impl!(RecordId, "record");

impl IdResolve for RecordId {
    fn resolve_id<'a>(&self, ast: &'a Ast) -> &'a str {
        ast.symbols
            .get(*ast.records.get(*self).expect("Record ID not found"))
            .expect("symbol not found")
    }

    fn get_name(&self) -> String {
        format!("rec_{}", self.as_u32())
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct FieldId(u32);
entity_impl!(FieldId, "field");

impl IdResolve for FieldId {
    fn resolve_id<'a>(&self, ast: &'a Ast) -> &'a str {
        ast.symbols
            .get(*ast.fields.get(*self).expect("Field ID not found"))
            .expect("symbol not found")
    }

    fn get_name(&self) -> String {
        format!("field_{}", self.as_u32())
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct TypeId(u32);
entity_impl!(TypeId, "typ");

#[derive(Debug)]
pub struct Ast {
    pub exprs: PrimaryMap<ExprId, Expr>,
    pub expr_lists: ListPool<ExprId>,

    pub commands: PrimaryMap<CommandId, Command>,
    pub command_lists: ListPool<CommandId>,

    empty_command: CommandId,

    pub symbols: PrimaryMap<Symbol, String>,
    symbol_lookup: HashMap<String, Symbol>,

    pub funcs: PrimaryMap<FuncId, Symbol>,
    func_lookup: HashMap<Symbol, FuncId>,

    pub values: PrimaryMap<ValueId, Symbol>,

    pub records: PrimaryMap<RecordId, Symbol>,
    record_lookup: HashMap<Symbol, RecordId>,

    pub fields: PrimaryMap<FieldId, Symbol>,
    field_lookup: HashMap<Symbol, FieldId>,
}

impl Ast {
    pub fn new() -> Self {
        let mut commands = PrimaryMap::new();
        let empty_command = commands.push(Command::Empty);
        Self {
            exprs: PrimaryMap::new(),
            expr_lists: ListPool::new(),
            commands,
            command_lists: ListPool::new(),
            empty_command,
            symbols: PrimaryMap::new(),
            symbol_lookup: HashMap::new(),
            funcs: PrimaryMap::new(),
            func_lookup: HashMap::new(),
            values: PrimaryMap::new(),
            records: PrimaryMap::new(),
            record_lookup: HashMap::new(),
            fields: PrimaryMap::new(),
            field_lookup: HashMap::new(),
        }
    }

    pub fn empty_command(&self) -> CommandId {
        self.empty_command
    }

    pub fn get_symbol(&mut self, name: &str) -> Symbol {
        if let Some(&sym) = self.symbol_lookup.get(name) {
            sym
        } else {
            let sym = self.symbols.push(name.to_string());
            self.symbol_lookup.insert(name.to_string(), sym);
            sym
        }
    }

    pub fn get_func(&mut self, symbol: Symbol) -> FuncId {
        if let Some(&func) = self.func_lookup.get(&symbol) {
            func
        } else {
            let func = self.funcs.push(symbol);
            self.func_lookup.insert(symbol, func);
            func
        }
    }

    pub fn get_record(&mut self, symbol: Symbol) -> RecordId {
        if let Some(&record) = self.record_lookup.get(&symbol) {
            record
        } else {
            let record = self.records.push(symbol);
            self.record_lookup.insert(symbol, record);
            record
        }
    }

    pub fn get_field(&mut self, symbol: Symbol) -> FieldId {
        if let Some(&field) = self.field_lookup.get(&symbol) {
            field
        } else {
            let field = self.fields.push(symbol);
            self.field_lookup.insert(symbol, field);
            field
        }
    }

    pub fn get_value(&mut self, symbol: Symbol) -> ValueId {
        self.values.push(symbol)
    }
}

#[derive(Debug, Default)]
pub struct TypeContext {
    pub types: PrimaryMap<TypeId, Type>,
    pub type_lists: ListPool<TypeId>,
    pub type_map: HashMap<TypeKey, TypeId>,

    pub expr_type_map: HashMap<ExprId, TypeId>,

    pub func_type_map: HashMap<FuncId, TypeId>,
    pub value_type_map: HashMap<ValueId, TypeId>,
}

#[derive(Debug)]
pub struct Context {
    pub ast: Ast,
    pub tcx: TypeContext,
}

impl Context {
    pub fn new() -> Self {
        Self {
            ast: Ast::new(),
            tcx: TypeContext::default(),
        }
    }
}

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
    Alias(RecordId),
    Array {
        element_type: TypeId,
        dims: Vec<DimSpec>,
        ports: usize,
    },
    StaticInt(i64),
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
        name: RecordId,
        fields: HashMap<FieldId, TypeId>,
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
    Alias(RecordId),
    Array {
        element_type: TypeId,
        dims: Vec<DimSpec>,
        ports: usize,
    },
    StaticInt(i64),
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
        name: RecordId,
        fields: Vec<(FieldId, TypeId)>,
    },
}

impl TypeContext {
    pub fn get_float(&mut self) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Float)
            .or_insert_with(|| self.types.push(Type::Float))
    }

    pub fn get_double(&mut self) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Double)
            .or_insert_with(|| self.types.push(Type::Double))
    }

    pub fn get_bool(&mut self) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Bool)
            .or_insert_with(|| self.types.push(Type::Bool))
    }

    pub fn get_bit(&mut self, length: usize, unsigned: bool) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Bit { length, unsigned })
            .or_insert_with(|| self.types.push(Type::Bit { length, unsigned }))
    }

    pub fn get_fixed(&mut self, length_total: usize, length_int: usize, unsigned: bool) -> TypeId {
        *self
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
    }

    pub fn get_alias(&mut self, name: RecordId) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Alias(name))
            .or_insert_with(|| self.types.push(Type::Alias(name)))
    }

    pub fn get_array(&mut self, element_type: TypeId, dims: Vec<DimSpec>, ports: usize) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Array {
                element_type,
                dims: dims.clone(),
                ports,
            })
            .or_insert_with(|| {
                self.types.push(Type::Array {
                    element_type,
                    dims,
                    ports,
                })
            })
    }

    pub fn get_static_int(&mut self, value: i64) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::StaticInt(value))
            .or_insert_with(|| self.types.push(Type::StaticInt(value)))
    }

    pub fn get_index(&mut self, static_: (i64, i64), dynamic: (i64, i64)) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Index { static_, dynamic })
            .or_insert_with(|| self.types.push(Type::Index { static_, dynamic }))
    }

    pub fn get_void(&mut self) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Void)
            .or_insert_with(|| self.types.push(Type::Void))
    }

    pub fn get_rational(&mut self, value: String) -> TypeId {
        *self
            .type_map
            .entry(TypeKey::Rational(value.clone()))
            .or_insert_with(|| self.types.push(Type::Rational(value)))
    }

    pub fn get_func(&mut self, args: Vec<TypeId>, ret: TypeId) -> TypeId {
        *self
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
    }

    pub fn get_rec_type(&mut self, name: RecordId, fields: HashMap<FieldId, TypeId>) -> TypeId {
        let mut field_key: Vec<_> = fields.iter().map(|(k, v)| (*k, *v)).collect();
        field_key.sort();

        *self
            .type_map
            .entry(TypeKey::RecType {
                name,
                fields: field_key,
            })
            .or_insert_with(|| self.types.push(Type::RecType { name, fields }))
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct DimSpec {
    pub length: usize,
    pub bank: usize,
}

#[derive(Debug, Clone, Copy)]
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
    Placeholder,
    Cast {
        expr: ExprId,
        ty: TypeId,
    },

    ArrayLiteral(EntityList<ExprId>),
    RecordLiteral(IndexMap<FieldId, ExprId>),

    RationalLiteral(String),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: ValueId,
        indices: EntityList<ExprId>,
    },
    RecordAccess {
        record: ExprId,
        field: FieldId,
    },

    Application {
        func: FuncId,
        args: EntityList<ExprId>,
    },

    Id(ValueId),

    BinOp {
        left: ExprId,
        op: InfixOp,
        right: ExprId,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

#[derive(Debug)]
pub struct ForRange {
    pub iter: ValueId,
    pub ty: Option<TypeId>,
    pub rev: bool,
    pub start: i64,
    pub end: i64,
    pub unroll: i64,
}

#[derive(Debug)]
pub enum Command {
    Empty,
    Block(CommandId),
    Par(Vec<CommandId>),
    Seq(Vec<CommandId>),
    Let {
        id: ValueId,
        ty: Option<TypeId>,
        value: Option<ExprId>,
    },
    Update {
        lhs: ExprId,
        op: AssignOp,
        rhs: ExprId,
    },
    View {
        id: ValueId,
        arr_id: ValueId,
        dims: Vec<View>,
    },
    Split {
        id: ValueId,
        arr_id: ValueId,
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
    pub id: ValueId,
    pub ty: TypeId,
}

#[derive(Debug)]
pub struct FuncSig {
    pub name: FuncId,
    pub args: Vec<Decl>,
    pub ret_ty: TypeId,
}

#[derive(Debug)]
pub enum Def {
    Func {
        sig: FuncSig,
        body: CommandId,
    },
    Record {
        name: RecordId,
        fields: HashMap<FieldId, TypeId>,
    },
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
    pub defs: Vec<Def>,
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
                match &context.ast.commands[cmd] {
                    Command::Par(cs) => flat.extend(cs),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            context.borrow().ast.empty_command()
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut context = context.borrow_mut();
            context.ast.commands.push(Command::Par(flat))
        }
    }

    pub fn smart_seq(cmds: Vec<CommandId>, context: &RefCell<Context>) -> CommandId {
        let mut flat = Vec::new();
        {
            let context = context.borrow();
            for cmd in cmds {
                match &context.ast.commands[cmd] {
                    Command::Seq(cs) => flat.extend(cs),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            context.borrow().ast.empty_command()
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut context = context.borrow_mut();
            context.ast.commands.push(Command::Seq(flat))
        }
    }
}
