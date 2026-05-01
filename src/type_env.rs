use std::{collections::HashMap, error::Error, fmt::Display};

use crate::{
    ast::{Context, DimSpec, Id, Type, TypeId},
    scoped_map::ScopedMap,
};

pub struct TypeEnv {
    type_map: ScopedMap<Id, TypeId>,
    typedefs: HashMap<Id, TypeId>,
    ret_type: Option<TypeId>,
}

#[derive(Debug)]
pub enum TypeEnvError {
    Unbound,
    AlreadyBound,
}

impl Display for TypeEnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeEnvError::Unbound => write!(f, "Type is unbound"),
            TypeEnvError::AlreadyBound => write!(f, "Type is already bound"),
        }
    }
}

impl Error for TypeEnvError {}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            type_map: ScopedMap::new(),
            typedefs: HashMap::new(),
            ret_type: None,
        }
    }

    pub fn add_type(&mut self, id: Id, type_id: TypeId) -> Result<(), TypeEnvError> {
        if self.typedefs.contains_key(&id) {
            Err(TypeEnvError::AlreadyBound)
        } else {
            self.typedefs.insert(id, type_id);
            Ok(())
        }
    }

    pub fn get_type(&self, id: &Id) -> Option<TypeId> {
        self.typedefs.get(id).copied()
    }

    pub fn resolve_type(&self, typ: TypeId, context: &mut Context) -> Option<TypeId> {
        enum TypeInfo {
            Func(Vec<TypeId>, TypeId),
            Array(TypeId, Vec<DimSpec>, usize),
        }

        let type_info = match context.types.get(typ)? {
            Type::Alias(id) => return self.get_type(id),
            Type::Func { args, ret } => {
                TypeInfo::Func(args.as_slice(&context.type_lists).to_vec(), *ret)
            }
            Type::Array {
                element_type,
                dims,
                ports,
            } => TypeInfo::Array(*element_type, dims.clone(), *ports),
            _ => return Some(typ),
        };

        match type_info {
            TypeInfo::Func(args, ret) => {
                let resolved_args: Vec<TypeId> = args
                    .into_iter()
                    .map(|arg| self.resolve_type(arg, context))
                    .collect::<Option<Vec<_>>>()?;
                let resolved_ret = self.resolve_type(ret, context)?;
                Some(context.get_func(resolved_args, resolved_ret))
            }
            TypeInfo::Array(element_type, dims, ports) => {
                let resolved_element_type = self.resolve_type(element_type, context)?;
                Some(context.get_array(resolved_element_type, dims, ports))
            }
        }
    }

    pub fn get(&self, id: &Id) -> Option<TypeId> {
        self.type_map.get(id).copied()
    }

    pub fn add(&mut self, id: Id, typ: TypeId, context: &mut Context) -> Result<(), TypeEnvError> {
        self.type_map
            .add(
                id,
                self.resolve_type(typ, context)
                    .expect("Type should resolve"),
            )
            .map_err(|_| TypeEnvError::AlreadyBound)
    }

    pub fn push_scope(&mut self) {
        self.type_map.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.type_map.pop_scope();
    }

    pub fn with_scope(&mut self, f: impl FnOnce(&mut Self)) {
        self.push_scope();
        f(self);
        self.pop_scope();
    }

    pub fn set_ret_type(&mut self, ret_type: TypeId) {
        self.ret_type = Some(ret_type);
    }

    pub fn get_ret_type(&self) -> Option<TypeId> {
        self.ret_type
    }
}
