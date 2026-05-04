use std::{collections::HashMap, hash::Hash};

use thiserror::Error;

pub struct ScopedMap<K, V> {
    scopes: Vec<HashMap<K, V>>,
}

#[derive(Debug, Error)]
pub enum ScopedMapError {
    #[error("Key already exists in the current scope")]
    KeyAlreadyExists,
    #[error("Key not found in any scope")]
    KeyNotFound,
}

impl<K: Eq + Hash, V> ScopedMap<K, V> {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn add(&mut self, key: K, value: V) -> Result<(), ScopedMapError> {
        match self.get(&key) {
            None => {
                self.scopes
                    .last_mut()
                    .expect("There should be at least one scope")
                    .insert(key, value);
                Ok(())
            }
            Some(_) => Err(ScopedMapError::KeyAlreadyExists),
        }
    }

    pub fn update(&mut self, key: K, value: V) -> Result<(), ScopedMapError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(&key) {
                scope.insert(key, value);
                return Ok(());
            }
        }
        Err(ScopedMapError::KeyNotFound)
    }

    pub fn add_shadow(&mut self, key: K, value: V) {
        self.scopes
            .last_mut()
            .expect("There should be at least one scope")
            .insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(key) {
                return Some(value);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{ScopedMap, ScopedMapError};

    #[test]
    fn add_makes_values_visible() {
        let mut map = ScopedMap::new();

        assert_eq!(map.get(&"answer"), None);
        map.add("answer", 42).unwrap();

        assert_eq!(map.get(&"answer"), Some(&42));
    }

    #[test]
    fn nested_scopes_hide_and_restore_outer_bindings() {
        let mut map = ScopedMap::new();

        map.add("x", 1).unwrap();
        map.push_scope();
        map.add_shadow("x", 2);
        map.add("y", 3).unwrap();

        assert_eq!(map.get(&"x"), Some(&2));
        assert_eq!(map.get(&"y"), Some(&3));

        map.pop_scope();

        assert_eq!(map.get(&"x"), Some(&1));
        assert_eq!(map.get(&"y"), None);
    }

    #[test]
    fn add_rejects_existing_visible_keys() {
        let mut map = ScopedMap::new();

        map.add("x", 1).unwrap();
        assert!(matches!(
            map.add("x", 2),
            Err(ScopedMapError::KeyAlreadyExists)
        ));

        map.push_scope();
        assert!(matches!(
            map.add("x", 3),
            Err(ScopedMapError::KeyAlreadyExists)
        ));
    }

    #[test]
    fn add_shadow_overwrites_current_scope_binding() {
        let mut map = ScopedMap::new();

        map.add_shadow("x", 1);
        map.add_shadow("x", 2);

        assert_eq!(map.get(&"x"), Some(&2));
    }

    #[test]
    fn update_changes_nearest_visible_binding() {
        let mut map = ScopedMap::new();

        map.add_shadow("x", 1);
        map.push_scope();
        map.add_shadow("x", 2);

        map.update("x", 3).unwrap();

        assert_eq!(map.get(&"x"), Some(&3));
        map.pop_scope();
        assert_eq!(map.get(&"x"), Some(&1));
    }

    #[test]
    fn update_reaches_outer_scope_when_not_shadowed() {
        let mut map = ScopedMap::new();

        map.add("x", 1).unwrap();
        map.push_scope();

        map.update("x", 2).unwrap();

        assert_eq!(map.get(&"x"), Some(&2));
        map.pop_scope();
        assert_eq!(map.get(&"x"), Some(&2));
    }

    #[test]
    fn update_reports_missing_keys() {
        let mut map: ScopedMap<&str, i32> = ScopedMap::new();

        assert!(matches!(
            map.update("missing", 1),
            Err(ScopedMapError::KeyNotFound)
        ));
    }
}
