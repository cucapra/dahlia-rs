use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::ast::{InfixOp, Type, TypeContext, TypeId};

fn bits_needed(n: i64) -> usize {
    match n {
        0 => 1,
        n if n > 0 => (u64::BITS - (n as u64).leading_zeros()) as usize,
        _ => (u64::BITS - n.unsigned_abs().leading_zeros()) as usize + 1,
    }
}

pub fn is_subtype(tid1: TypeId, tid2: TypeId, tcx: &TypeContext) -> bool {
    let t1 = tcx
        .types
        .get(tid1)
        .expect("Type ID not found in context");
    let t2 = tcx
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
        (Type::Rational(_), Type::Float | Type::Double) => true,
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

fn eval_op<T>(op: InfixOp, v1: T, v2: T) -> Option<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>,
{
    match op {
        InfixOp::Add => Some(v1 + v2),
        InfixOp::Sub => Some(v1 - v2),
        InfixOp::Mod => Some(v1 % v2),
        InfixOp::Mul => Some(v1 * v2),
        InfixOp::Div => Some(v1 / v2),
        _ => None,
    }
}

fn join_of_helper(
    tid1: TypeId,
    tid2: TypeId,
    op: InfixOp,
    tcx: &mut TypeContext,
) -> Option<TypeId> {
    let t1 = tcx
        .types
        .get(tid1)
        .expect("Type ID not found in context");
    let t2 = tcx
        .types
        .get(tid2)
        .expect("Type ID not found in context");

    match (t1, t2) {
        (Type::StaticInt(v1), Type::StaticInt(v2)) => {
            if let Some(val) = eval_op(op, *v1, *v2) {
                Some(tcx.get_static_int(val))
            } else {
                Some(tcx.get_bit(bits_needed(*v1).max(bits_needed(*v2)), false))
            }
        }
        (Type::Rational(v1), Type::Rational(v2)) => {
            let v1: f64 = v1.parse().expect("Invalid rational number");
            let v2: f64 = v2.parse().expect("Invalid rational number");
            if let Some(val) = eval_op(op, v1, v2) {
                Some(tcx.get_rational(val.to_string()))
            } else {
                if bits_needed(v1 as i64) > bits_needed(v2 as i64) {
                    Some(tcx.get_rational(v1.to_string()))
                } else {
                    Some(tcx.get_rational(v2.to_string()))
                }
            }
        }
        (
            Type::Bit {
                length: s1,
                unsigned: un1,
            },
            Type::Bit {
                length: s2,
                unsigned: un2,
            },
        ) => {
            if un1 == un2 {
                Some(tcx.get_bit(*s1.max(s2), *un1))
            } else {
                None
            }
        }
        (
            Type::Bit {
                length: s,
                unsigned: un,
            },
            Type::StaticInt(v),
        ) => Some(tcx.get_bit(bits_needed(*v).max(*s), *un)),
        (Type::StaticInt(v), Type::Index { static_, dynamic }) => {
            let max_val = static_.1 * dynamic.1 - 1;
            Some(tcx.get_bit(bits_needed(*v.max(&max_val)), false))
        }
        (Type::Bit { .. }, Type::Index { .. }) => Some(tid1),
        (Type::Float | Type::Rational(..), Type::Double) => Some(tcx.get_double()),
        (Type::Rational(..), Type::Float) => Some(tcx.get_float()),
        (
            Type::Rational(v1),
            Type::Fixed {
                length_total: t2,
                length_int: i2,
                unsigned: un2,
            },
        ) => {
            let v1 = v1.parse::<f64>().expect("Invalid rational number") as i64;
            Some(tcx.get_fixed(
                *i2.max(&bits_needed(v1)) + t2 - i2,
                *i2.max(&bits_needed(v1)),
                *un2,
            ))
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
        ) => {
            if un1 == un2 {
                Some(tcx.get_fixed(i1.max(i2) + (t1 - i1).max(t2 - i2), *i1.max(i2), *un1))
            } else {
                None
            }
        }
        (
            Type::Index {
                static_: s1,
                dynamic: d1,
            },
            Type::Index {
                static_: s2,
                dynamic: d2,
            },
        ) => {
            let max_val1 = s1.1 * d1.1 - 1;
            let max_val2 = s2.1 * d2.1 - 1;
            Some(tcx.get_bit(bits_needed(max_val1).max(bits_needed(max_val2)), false))
        }
        (_, _) => {
            if tid1 == tid2 {
                Some(tid1)
            } else {
                None
            }
        }
    }
}

pub fn join_of(tid1: TypeId, tid2: TypeId, op: InfixOp, tcx: &mut TypeContext) -> Option<TypeId> {
    if let Some(join) = join_of_helper(tid1, tid2, op.clone(), tcx) {
        Some(join)
    } else {
        join_of_helper(tid2, tid1, op, tcx)
    }
}

pub fn safe_cast(tid_from: TypeId, tid_to: TypeId, tcx: &TypeContext) -> bool {
    let tfrom = tcx
        .types
        .get(tid_from)
        .expect("Type ID not found in context");
    let tto = tcx
        .types
        .get(tid_to)
        .expect("Type ID not found in context");

    match (tfrom, tto) {
        (Type::StaticInt(..) | Type::Index { .. } | Type::Bit { .. }, Type::Bit { .. }) => {
            is_subtype(tid_from, tid_to, tcx)
        }
        (Type::Float | Type::Double | Type::Rational(_), Type::Bit { .. }) => false,
        (Type::StaticInt(..) | Type::Index { .. } | Type::Bit { .. }, Type::Float) => true,
        (Type::Float, Type::Double) => true,
        (Type::Rational(..), Type::Double) => true,
        (
            Type::Bit {
                length: i1,
                unsigned: un1,
            },
            Type::Fixed {
                length_int: i2,
                unsigned: un2,
                ..
            },
        ) => un1 == un2 && i1 <= i2,
        _ => tid_from == tid_to, // can directly compare TypeIds as they are interned
    }
}
