use crate::ast::{Context, Type, TypeId};

fn bits_needed(n: i64) -> usize {
    match n {
        0 => 1,
        n if n > 0 => (u64::BITS - (n as u64).leading_zeros()) as usize,
        _ => (u64::BITS - n.unsigned_abs().leading_zeros()) as usize + 1,
    }
}

pub fn is_subtype(tid1: TypeId, tid2: TypeId, context: &Context) -> bool {
    let t1 = context
        .types
        .get(tid1)
        .expect("Type ID not found in context");
    let t2 = context
        .types
        .get(tid2)
        .expect("Type ID not found in context");

    match (t1, t2) {
        (
            Type::Bit {
                length: v1,
                unsigned: un1,
            },
            Type::Bit {
                length: v2,
                unsigned: un2,
            },
        ) => un1 == un2 && v1 <= v2,
        (
            Type::StaticInt(v1),
            Type::Bit {
                length: v2,
                unsigned: un2,
            },
        ) => (((*v1 < 0) && !un2) || *v1 >= 0) && bits_needed(*v1) <= *v2,
        (Type::Index { .. }, Type::Bit { .. }) => true,
        (Type::StaticInt(_), Type::Index { .. }) => true,
        (
            Type::Array {
                element_type: tsub,
                dims: sub_dims,
                ports: p1,
            },
            Type::Array {
                element_type: tsup,
                dims: sup_dims,
                ports: p2,
            },
        ) => tsup == tsub && sub_dims == sup_dims && p1 == p2,
        (Type::Float, Type::Double) => true,
        (Type::Rational(_), Type::Float) => true,
        (Type::Rational(_), Type::Double) => true,
        (
            Type::Rational(v1),
            Type::Fixed {
                length_total: _,
                length_int: i2,
                unsigned: un2,
            },
        ) => {
            let v1: f64 = v1.parse().expect("Invalid rational number");
            ((v1 < 0.0 && !un2) || v1 >= 0.0) && bits_needed(v1 as i64) <= *i2
        }
        (
            Type::Fixed {
                length_total: t1,
                length_int: i1,
                unsigned: un1,
            },
            Type::Fixed {
                length_total: t2,
                length_int: i2,
                unsigned: un2,
            },
        ) => un1 == un2 && i1 <= i2 && t1 - i1 <= t2 - i2,
        _ => tid1 == tid2, // can directly compare TypeIds as they are interned
    }
}
