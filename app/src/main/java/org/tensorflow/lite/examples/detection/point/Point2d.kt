package org.tensorflow.lite.examples.detection.point

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

class Point2d: Tuple2d, java.io.Serializable{
    /**
     * Constructs and initializes a Point3d from the specified xyz coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     */
    constructor(x: Float, y: Float): super(x, y)

    constructor(x: Float, y: Float, confidence: Float): super(x, y, confidence)

    /**
     * Constructs and initializes a Point2d from the array of length 2.
     * @param p the array of length 2 containing xyz in order
     */
    constructor(p: FloatArray) :super(p)


    /**
     * Constructs and initializes a Point3d from the specified Point3d.
     * @param p1 the Point3d containing the initialization x y data
     */
    constructor(p1: Point2d): super(p1 as Tuple2d)

    /**
     * Constructs and initializes a Point3d from the specified Tuple3d.
     * @param t1 the Tuple3d containing the initialization x y data
     */
    constructor(t1: Tuple2d): super(t1)


    /**
     * Constructs and initializes a Point3d to (0,0,0).
     */
    constructor(): super()

    /**
     * Returns the square of the distance between this point and point p1.
     * @param p1 the other point
     * @return the square of the distance
     */
    fun distanceSquared(p1: Point2d): Float {
        val dx: Float = x - p1.x
        val dy: Float = y - p1.y
        return dx * dx + dy * dy
    }


    /**
     * Returns the distance between this point and point p1.
     * @param p1 the other point
     * @return the distance
     */
    fun distance(p1: Point2d): Float {
        val dx: Float = x - p1.x
        val dy: Float = y - p1.y
        return sqrt(dx * dx + dy * dy)
    }


    /**
     * Computes the L-1 (Manhattan) distance between this point and
     * point p1.  The L-1 distance is equal to:
     * abs(x1-x2) + abs(y1-y2).
     * @param p1 the other point
     * @return  the L-1 distance
     */
    fun distanceL1(p1: Point2d): Float {
        return (abs(x - p1.x) + abs(y - p1.y))
    }

    /**
     * Computes the L-infinite distance between this point and
     * point p1.  The L-infinite distance is equal to
     * MAX[abs(x1-x2), abs(y1-y2)].
     * @param p1 the other point
     * @return  the L-infinite distance
     */
    fun distanceLinf(p1: Point2d): Float {
        return max(abs(x - p1.x), abs(y - p1.y))
    }
}