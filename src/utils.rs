pub fn bits_needed(n: i64) -> usize {
    match n {
        0 => 1,
        n if n > 0 => (u64::BITS - (n as u64).leading_zeros()) as usize,
        _ => (u64::BITS - n.unsigned_abs().leading_zeros()) as usize + 1,
    }
}
