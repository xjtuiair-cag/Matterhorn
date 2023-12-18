#include <cmath>
#include <iostream>
#include <stdlib.h>
#include "base.h"

/*
取绝对值。
Args:
    base (float): 原数值
Returns:
    res (float): 原数值的绝对值
*/
__device__ float absf(float base) {
    return base >= 0.0f ? base : -base;
}

/*
取符号。
Args:
    base (float): 原数值
Returns:
    res (float): 原数值的符号（+1，0，-1）
*/
__device__ float sgnf(float base) {
    return base != 0.0f ? (base > 0.0f ? 1.0f : -1.0f) : 0.0f;
}

/*
逻辑非。
Args:
    base (float): 原数值
Returns:
    res (float): 逻辑运算结果（1，0）
*/
__device__ float logical_notf(float base) {
    return base ? 0.0f : 1.0f;
}

/*
等于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float eqf(float base, float comp) {
    return base == comp ? 1.0f : 0.0f;
}

/*
不等于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float nef(float base, float comp) {
    return base != comp ? 1.0f : 0.0f;
}

/*
小于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float ltf(float base, float comp) {
    return base < comp ? 1.0f : 0.0f;
}

/*
小于等于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float lef(float base, float comp) {
    return base <= comp ? 1.0f : 0.0f;
}

/*
大于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float gtf(float base, float comp) {
    return base > comp ? 1.0f : 0.0f;
}

/*
大于等于比较。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 比较结果
*/
__device__ float gef(float base, float comp) {
    return base >= comp ? 1.0f : 0.0f;
}

/*
逻辑与。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 逻辑运算结果（1，0）
*/
__device__ float logical_andf(float base, float comp) {
    return base && comp ? 1.0f : 0.0f;
}

/*
逻辑或。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 逻辑运算结果（1，0）
*/
__device__ float logical_orf(float base, float comp) {
    return base || comp ? 1.0f : 0.0f;
}

/*
逻辑异或。
Args:
    base (float): 原数值
    comp (float): 待比较的数值
Returns:
    res (float): 逻辑运算结果（1，0）
*/
__device__ float logical_xorf(float base, float comp) {
    return ((base || comp) && !(base && comp)) ? 1.0f : 0.0f;
}

/*
区间函数，不取两侧边界，相当于判断数值是否在$(min,max)$内。
Args:
    base (float): 原数值
    min (float): 最小值
    max (float): 最大值
Returns:
    res (float): 取区间结果（1，0）
*/
__device__ float winnbf(float base, float min, float max) {
    return ((base > min) && (base < max)) ? 1.0f : 0.0f;
}

/*
区间函数，取下界，相当于判断数值是否在$[min,max)$内。
Args:
    base (float): 原数值
    min (float): 最小值
    max (float): 最大值
Returns:
    res (float): 取区间结果（1，0）
*/
__device__ float winlbf(float base, float min, float max) {
    return ((base >= min) && (base < max)) ? 1.0f : 0.0f;
}

/*
区间函数，取上界，相当于判断数值是否在$(min,max]$内。
Args:
    base (float): 原数值
    min (float): 最小值
    max (float): 最大值
Returns:
    res (float): 取区间结果（1，0）
*/
__device__ float winrbf(float base, float min, float max) {
    return ((base > min) && (base <= max)) ? 1.0f : 0.0f;
}

/*
区间函数，取两侧边界，相当于判断数值是否在$[min,max]$内。
Args:
    base (float): 原数值
    min (float): 最小值
    max (float): 最大值
Returns:
    res (float): 取区间结果（1，0）
*/
__device__ float winbf(float base, float min, float max) {
    return ((base >= min) && (base <= max)) ? 1.0f : 0.0f;
}

/*
钳函数，如果小于min，取min；如果大于max，取max；否则取原数值。
Args:
    base (float): 原数值
    min (float): 最小值
    max (float): 最大值
Returns:
    res (float): 钳函数结果
*/
__device__ float clampf(float base, float min, float max) {
    return base > min ? (base < max ? base : max) : min;
}