use crate::{
    ast::{Ast, Symbol, ValueId},
    errors::ResolveError,
    scoped_map::ScopedMap,
};

pub struct ResolverEnv {
    pub symbol_map: ScopedMap<Symbol, ValueId>,
}

impl ResolverEnv {
    pub fn new() -> Self {
        Self {
            symbol_map: ScopedMap::new(),
        }
    }

    pub fn add(&mut self, id: ValueId, ast: &Ast) -> Result<(), ResolveError> {
        self.symbol_map
            .add(ast.values[id], id)
            .map_err(|_| ResolveError::AlreadyBound)
    }

    pub fn get(&self, id: &ValueId, ast: &Ast) -> Option<ValueId> {
        self.symbol_map.get(&ast.values[*id]).copied()
    }

    pub fn with_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.symbol_map.push_scope();
        let result = f(self);
        self.symbol_map.pop_scope();
        result
    }
}
