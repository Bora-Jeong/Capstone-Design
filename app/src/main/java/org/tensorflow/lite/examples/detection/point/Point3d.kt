package org.tensorflow.lite.examples.detection.point

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

class Point3d : Tuple3d, java.io.Serializable {

    /**
     * Constructs and initializes a Point3d from the specified xyz coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     * @param z the z coordinate
     */
    constructor(x: Float, y: Float, z: Float): super(x, y, z)

    constructor(x: Float, y: Float, z: Float, confidence: Float): super(x, y, z, confidence)

    /**
     * Constructs and initializes a Point3d from the array of length 3.
     * @param p the array of length 3 containing xyz in order
     */
    constructor(p: FloatArray) {
        Tuple3d(p)
    }


    /**
     * Constructs and initializes a Point3d from the specified Point3d.
     * @param p1 the Point3d containing the initialization x y z data
     */
    constructor(p1: Point3d) {
        Tuple3d((p1 as Tuple3d))
    }

    /**
     * Constructs and initializes a Point3d from the specified Tuple3d.
     * @param t1 the Tuple3d containing the initialization x y z data
     */
    constructor(t1: Tuple3d) {
        Tuple3d(t1)
    }


    /**
     * Constructs and initializes a Point3d to (0,0,0).
     */
    constructor() {
        Tuple3d()
    }

    /**
     * Returns the square of the distance between this point and point p1.
     * @param p1 the other point
     * @return the square of the distance
     */
    fun distanceSquared(p1: Point3d): Float {
        val dx: Float = x - p1.x
        val dy: Float = y - p1.y
        val dz: Float = z - p1.z
        return dx * dx + dy * dy + dz * dz
    }


    /**
     * Returns the distance between this point and point p1.
     * @param p1 the other point
     * @return the distance
     */
    fun distance(p1: Point3d): Float {
        val dx: Float = x - p1.x
        val dy: Float = y - p1.y
        val dz: Float = z - p1.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }


    /**
     * Computes the L-1 (Manhattan) distance between this point and
     * point p1.  The L-1 distance is equal to:
     * abs(x1-x2) + abs(y1-y2) + abs(z1-z2).
     * @param p1 the other point
     * @return  the L-1 distance
     */
    fun distanceL1(p1: Point3d): Float {
        return (abs(x - p1.x) + abs(y - p1.y) + abs(z - p1.z))
    }


    /**
     * Computes the L-infinite distance between this point and
     * point p1.  The L-infinite distance is equal to
     * MAX[abs(x1-x2), abs(y1-y2), abs(z1-z2)].
     * @param p1 the other point
     * @return  the L-infinite distance
     */
    fun distanceLinf(p1: Point3d): Float {
        val tmp: Float = max(abs(x - p1.x), abs(y - p1.y))
        return max(tmp, abs(z - p1.z))
    }


    /**
     * Multiplies each of the x,y,z components of the Point4d parameter
     * by 1/w and places the projected values into this point.
     * @param  p1  the source Point4d, which is not modified
     */
    /*
    fun project(p1: Point4d) {
        val oneOw: Double
        oneOw = 1 / p1.w
        x = p1.x * oneOw
        y = p1.y * oneOw
        z = p1.z * oneOw
    }
     */
}