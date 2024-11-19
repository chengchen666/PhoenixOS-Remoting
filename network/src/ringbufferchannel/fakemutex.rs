#![expect(dead_code)]

use std::cell::UnsafeCell;

pub struct FakeMutex<T> {
    inner: UnsafeCell<T>,
}

impl<T> FakeMutex<T> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner: UnsafeCell::new(inner),
        }
    }

    pub fn lock(&self) -> FakeMutexGuard<T> {
        FakeMutexGuard { mutex: self }
    }
}

unsafe impl<T: Send> Sync for FakeMutex<T> {}

pub struct FakeMutexGuard<'a, T> {
    mutex: &'a FakeMutex<T>,
}

impl<'a, T> FakeMutexGuard<'a, T> {
    pub fn unwrap(self) -> &'a mut T {
        unsafe { &mut *self.mutex.inner.get() }
    }
}

impl<T> std::ops::Deref for FakeMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.inner.get() }
    }
}

impl<T> std::ops::DerefMut for FakeMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.inner.get() }
    }
}

impl<T> Drop for FakeMutexGuard<'_, T> {
    fn drop(&mut self) {
        // No actual unlocking needed
    }
}
