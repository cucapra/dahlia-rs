use std::collections::HashMap;

use crate::{
    ast::{FuncId, RecordId, Type, TypeContext, TypeId, ValueId},
    errors::TypeEnvError,
};

pub struct TypeEnv {
    val_map: HashMap<ValueId, TypeId>,
    record_map: HashMap<RecordId, TypeId>,
    func_map: HashMap<FuncId, TypeId>,
    ret_type: Option<TypeId>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            val_map: HashMap::new(),
            record_map: HashMap::new(),
            func_map: HashMap::new(),
            ret_type: None,
        }
    }

    pub fn add_func(&mut self, id: FuncId, type_id: TypeId) -> Result<(), TypeEnvError> {
        if self.func_map.contains_key(&id) {
            Err(TypeEnvError::AlreadyBound)
        } else {
            self.func_map.insert(id, type_id);
            Ok(())
        }
    }

    pub fn add_record(&mut self, id: RecordId, type_id: TypeId) -> Result<(), TypeEnvError> {
        if self.record_map.contains_key(&id) {
            Err(TypeEnvError::AlreadyBound)
        } else {
            self.record_map.insert(id, type_id);
            Ok(())
        }
    }

    pub fn get_func(&self, id: &FuncId) -> Option<TypeId> {
        self.func_map.get(id).copied()
    }

    pub fn get_record(&self, id: &RecordId) -> Option<TypeId> {
        self.record_map.get(id).copied()
    }

    pub fn resolve_type(
        &self,
        type_id: TypeId,
        tcx: &mut TypeContext,
    ) -> Result<TypeId, TypeEnvError> {
        let r#type = match tcx.types.get(type_id).expect("Type ID not found") {
            Type::Alias(id) => {
                return self.get_record(id).ok_or(TypeEnvError::UnknownAlias);
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

    pub fn get(&self, id: &ValueId) -> Option<TypeId> {
        self.val_map.get(id).copied()
    }

    pub fn add(
        &mut self,
        id: ValueId,
        type_id: TypeId,
        tcx: &mut TypeContext,
    ) -> Result<(), TypeEnvError> {
        if self.val_map.contains_key(&id) {
            Err(TypeEnvError::AlreadyBound)
        } else {
            self.val_map.insert(id, self.resolve_type(type_id, tcx)?);
            Ok(())
        }
    }

    pub fn set_ret_type(&mut self, ret_type: TypeId) {
        self.ret_type = Some(ret_type);
    }

    pub fn get_ret_type(&self) -> Option<TypeId> {
        self.ret_type
    }
}
