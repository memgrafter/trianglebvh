//! Partial sort (selection) with the element order of GNU libstdc++'s std::nth_element.
//!
//! Probes show that cubvh's tree is built with a median selection whose effect on the order of elements (also of
//! equal keys) is that of libstdc++'s introselect: median-of-three pivot moved to the front, unguarded Hoare
//! partition, heap selection when the depth limit (2 * floor(log2(n))) runs out, insertion sort below four elements.
//! This module implements that published algorithm (D. Musser, "Introspective Sorting and Selection Algorithms",
//! 1997; C. A. R. Hoare's partition) so the resulting order is the same. Only the order matters here: the tie between
//! equal distances is decided by where a triangle ends up.

/// Reorder `v` so that v[nth] is the element a full sort would put there, everything before it is not greater and
/// everything after it is not smaller (by `less`), with libstdc++'s exact resulting order.
pub fn nth_element<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], nth: usize, less: &F) {
    let n = v.len();
    if n == 0 || nth >= n {
        return;
    }
    let mut first = 0usize;
    let mut last = n;
    let mut depth = 2 * lg(n);
    while last - first > 3 {
        if depth == 0 {
            heap_select(&mut v[first..last], nth + 1 - first, less);
            v.swap(first, nth);
            return;
        }
        depth -= 1;
        let cut = partition_pivot(&mut v[first..last], less) + first;
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(&mut v[first..last], less);
}

fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// Median of v[1], v[mid], v[len - 1] moved to v[0], then the partition of v[1..] around v[0]; returns the cut.
fn partition_pivot<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) -> usize {
    let len = v.len();
    let mid = len / 2;
    move_median_to_first(v, 0, 1, mid, len - 1, less);
    unguarded_partition(v, 1, len, 0, less)
}

fn move_median_to_first<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], result: usize, a: usize, b: usize, c: usize, less: &F) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(result, b);
        } else if less(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if less(&v[a], &v[c]) {
        v.swap(result, a);
    } else if less(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], mut first: usize, mut last: usize, pivot: usize, less: &F) -> usize {
    loop {
        while less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn insertion_sort<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    for i in 1..v.len() {
        let val = v[i];
        if less(&val, &v[0]) {
            v.copy_within(0..i, 1);
            v[0] = val;
        } else {
            let mut j = i;
            while less(&val, &v[j - 1]) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = val;
        }
    }
}

/// The `middle` smallest elements of v go to the front (a max-heap there), as libstdc++'s __heap_select.
fn heap_select<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], middle: usize, less: &F) {
    make_heap(&mut v[..middle], less);
    for i in middle..v.len() {
        if less(&v[i], &v[0]) {
            // pop_heap(first, middle, i): the top goes to v[i], v[i]'s value sifts into the heap
            let val = v[i];
            v[i] = v[0];
            adjust_heap(&mut v[..middle], 0, val, less);
        }
    }
}

fn make_heap<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    let len = v.len();
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let val = v[parent];
        adjust_heap(v, parent, val, less);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn adjust_heap<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], mut hole: usize, val: T, less: &F) {
    let len = v.len();
    let top = hole;
    let mut second = hole;
    while second < (len.wrapping_sub(1)) / 2 && len > 0 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len >= 2 && (len & 1) == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    // push_heap(first, hole, top, val)
    while hole > top {
        let parent = (hole - 1) / 2;
        if !less(&v[parent], &val) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = val;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(v: &[i64], nth: usize) {
        let mut s = v.to_vec();
        s.sort();
        let mut w = v.to_vec();
        nth_element(&mut w, nth, &|a: &i64, b: &i64| a < b);
        assert_eq!(w[nth], s[nth]);
        assert!(w[..nth].iter().all(|x| *x <= w[nth]));
        assert!(w[nth + 1..].iter().all(|x| *x >= w[nth]));
    }

    #[test]
    fn selects() {
        let mut x: u64 = 1;
        for n in 1..300usize {
            let v: Vec<i64> = (0..n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((x >> 33) % 17) as i64 }).collect();
            for nth in [0, n / 2, n - 1] {
                check(&v, nth);
            }
        }
    }

    #[test]
    fn heap_select_selects() {
        // an organ-pipe input drives some partitions deep; heap_select must still select correctly
        for n in 4..200usize {
            let mut v: Vec<i64> = (0..n as i64).map(|i| if i < n as i64 / 2 { i } else { n as i64 - i }).collect();
            let mut s = v.clone();
            s.sort();
            heap_select(&mut v, n / 2 + 1, &|a: &i64, b: &i64| a < b);
            let mut head = v[..n / 2 + 1].to_vec();
            head.sort();
            assert_eq!(head, s[..n / 2 + 1].to_vec());
        }
    }
}
