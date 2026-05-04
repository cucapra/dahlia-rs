use std::{collections::HashMap, error::Error, fmt::Display};

use crate::{
    ast::{Id, Type, TypeContext, TypeId},
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
    UnknownAlias,
}

impl Display for TypeEnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeEnvError::Unbound => write!(f, "Type is unbound"),
            TypeEnvError::AlreadyBound => write!(f, "Type is already bound"),
            TypeEnvError::UnknownAlias => write!(f, "Unknown alias"),
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

    pub fn resolve_type(
        &self,
        type_id: TypeId,
        tcx: &mut TypeContext,
    ) -> Result<TypeId, TypeEnvError> {
        let r#type = match tcx.types.get(type_id).expect("Type ID not found") {
            Type::Alias(id) => {
                return self.get_type(id).ok_or(TypeEnvError::UnknownAlias);
            }
            func @ Type::Func { .. } => func.clone(),
            arr @ Type::Array { .. } => arr.clone(),
            _ => return Ok(type_id),
        };

        match r#type {
            Type::Func { args, ret } => {
                let resolved_args: Vec<TypeId> =
                    args.as_slice(&tcx.type_lists).iter().copied().collect();
                let resolved_args = resolved_args
                    .into_iter()
                    .map(|arg| self.resolve_type(arg, tcx))
                    .collect::<Result<Vec<_>, _>>()?;
                let resolved_ret = self.resolve_type(ret, tcx)?;
                Ok(tcx.get_func(resolved_args, resolved_ret))
            }
            Type::Array {
                element_type,
                dims,
                ports,
            } => {
                let resolved_element_type = self.resolve_type(element_type, tcx)?;
                Ok(tcx.get_array(resolved_element_type, dims, ports))
            }
            _ => unreachable!(),
        }
    }

    pub fn get(&self, id: &Id) -> Option<TypeId> {
        self.type_map.get(id).copied()
    }

    pub fn add(
        &mut self,
        id: Id,
        type_id: TypeId,
        tcx: &mut TypeContext,
    ) -> Result<(), TypeEnvError> {
        self.type_map
            .add(id, self.resolve_type(type_id, tcx)?)
            .map_err(|_| TypeEnvError::AlreadyBound)
    }

    pub fn push_scope(&mut self) {
        self.type_map.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.type_map.pop_scope();
    }

    pub fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    pub fn set_ret_type(&mut self, ret_type: TypeId) {
        self.ret_type = Some(ret_type);
    }

    pub fn get_ret_type(&self) -> Option<TypeId> {
        self.ret_type
    }
}
