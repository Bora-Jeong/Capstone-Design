package org.tensorflow.lite.examples.detection.point

import kotlin.math.abs

open class Tuple3d : java.io.Serializable, Cloneable{
    /**
     * The x coordinate.
     */
    var x = 0.0F

    /**
     * The y coordinate.
     */
    var y = 0.0F

    /**
     * The z coordinate.
     */
    var z = 0.0F

    var confidence = 0.0F


    /**
     * Constructs and initializes a Tuple3d from the specified xyz coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     * @param z the z coordinate
     */
    constructor(x: Float, y: Float, z: Float) {
        this.x = x
        this.y = y
        this.z = z
    }

    constructor(x: Float, y: Float, z: Float, confidence: Float) {
        this.x = x
        this.y = y
        this.z = z
        this.confidence = confidence
    }

    /**
     * Constructs and initializes a Tuple3d from the array of length 3.
     * @param t the array of length 3 containing xyz in order
     */
    constructor(t: FloatArray) {
        x = t[0]
        y = t[1]
        z = t[2]
        if (t.size > 3) {
            confidence = t[3]
        }
    }

    /**
     * Constructs and initializes a Tuple3d from the specified Tuple3d.
     * @param t1 the Tuple3d containing the initialization x y z data
     */
    constructor(t1: Tuple3d) {
        x = t1.x
        y = t1.y
        z = t1.z
        confidence = t1.confidence
    }

    /**
     * Constructs and initializes a Tuple3d to (0,0,0).
     */
    constructor() {
        x = 0.0F
        y = 0.0F
        z = 0.0F
        confidence = 0.0F
    }

    /**
     * Sets the value of this tuple to the specified xyz coordinates.
     * @param x the x coordinate
     * @param y the y coordinate
     * @param z the z coordinate
     */
    operator fun set(x: Float, y: Float, z: Float) {
        this.x = x
        this.y = y
        this.z = z
    }

    operator fun set(x: Float, y: Float, z: Float, confidence: Float) {
        this.x = x
        this.y = y
        this.z = z
        this.confidence = confidence
    }

    /**
     * Sets the value of this tuple to the value of the xyz coordinates
     * located in the array of length 3.
     * @param t the array of length 3 containing xyz in order
     */
    fun set(t: FloatArray) {
        x = t[0]
        y = t[1]
        z = t[2]
        if(t.size > 3) {
            confidence = t[3]
        }
    }

    /**
     * Sets the value of this tuple to the value of tuple t1.
     * @param t1 the tuple to be copied
     */
    fun set(t1: Tuple3d) {
        x = t1.x
        y = t1.y
        z = t1.z
        confidence = t1.confidence
    }

    /**
     * Copies the x,y,z coordinates of this tuple into the array t
     * of length 3.
     * @param t  the target array
     */
    operator fun get(t: FloatArray) {
        t[0] = x
        t[1] = y
        t[2] = z
        if(t.size > 3){
            t[3] = confidence
        }
    }


    /**
     * Copies the x,y,z coordinates of this tuple into the tuple t.
     * @param t  the Tuple3d object into which the values of this object are copied
     */
    operator fun get(t: Tuple3d) {
        t.x = x
        t.y = y
        t.z = z
        t.confidence = confidence
    }


    /**
     * Sets the value of this tuple to the sum of tuples t1 and t2.
     * @param t1 the first tuple
     * @param t2 the second tuple
     */
    fun add(t1: Tuple3d, t2: Tuple3d) {
        x = t1.x + t2.x
        y = t1.y + t2.y
        z = t1.z + t2.z
    }


    /**
     * Sets the value of this tuple to the sum of itself and t1.
     * @param t1 the other tuple
     */
    fun add(t1: Tuple3d) {
        x += t1.x
        y += t1.y
        z += t1.z
    }

    /**
     * Sets the value of this tuple to the difference of tuples
     * t1 and t2 (this = t1 - t2).
     * @param t1 the first tuple
     * @param t2 the second tuple
     */
    fun sub(t1: Tuple3d, t2: Tuple3d) {
        x = t1.x - t2.x
        y = t1.y - t2.y
        z = t1.z - t2.z
    }

    /**
     * Sets the value of this tuple to the difference
     * of itself and t1 (this = this - t1).
     * @param t1 the other tuple
     */
    fun sub(t1: Tuple3d) {
        x -= t1.x
        y -= t1.y
        z -= t1.z
    }


    /**
     * Sets the value of this tuple to the negation of tuple t1.
     * @param t1 the source tuple
     */
    fun negate(t1: Tuple3d) {
        x = -t1.x
        y = -t1.y
        z = -t1.z
    }


    /**
     * Negates the value of this tuple in place.
     */
    fun negate() {
        x = -x
        y = -y
        z = -z
    }


    /**
     * Sets the value of this tuple to the scalar multiplication
     * of tuple t1.
     * @param s the scalar value
     * @param t1 the source tuple
     */
    fun scale(s: Float, t1: Tuple3d) {
        x = s * t1.x
        y = s * t1.y
        z = s * t1.z
    }


    /**
     * Sets the value of this tuple to the scalar multiplication
     * of itself.
     * @param s the scalar value
     */
    fun scale(s: Float) {
        x *= s
        y *= s
        z *= s
    }


    /**
     * Sets the value of this tuple to the scalar multiplication
     * of tuple t1 and then adds tuple t2 (this = s*t1 + t2).
     * @param s the scalar value
     * @param t1 the tuple to be multipled
     * @param t2 the tuple to be added
     */
    fun scaleAdd(s: Float, t1: Tuple3d, t2: Tuple3d) {
        x = s * t1.x + t2.x
        y = s * t1.y + t2.y
        z = s * t1.z + t2.z
    }

    /**
     * Sets the value of this tuple to the scalar multiplication
     * of itself and then adds tuple t1 (this = s*this + t1).
     * @param s the scalar value
     * @param t1 the tuple to be added
     */
    fun scaleAdd(s: Float, t1: Tuple3d) {
        x = s * x + t1.x
        y = s * y + t1.y
        z = s * z + t1.z
    }


    /**
     * Returns a string that contains the values of this Tuple3d.
     * The form is (x,y,z).
     * @return the String representation
     */
    override fun toString(): String {
        return "($x, $y, $z)"
    }


    /**
     * Returns a hash code value based on the data values in this
     * object.  Two different Tuple3d objects with identical data values
     * (i.e., Tuple3d.equals returns true) will return the same hash
     * code value.  Two objects with different data members may return the
     * same hash value, although this is not likely.
     * @return the integer hash code value
     */
    override fun hashCode(): Int {
        var bits = 1L
        bits *= 31L
        bits += x.hashCode().toLong()
        bits *= 31L
        bits += y.hashCode().toLong()
        bits *= 31L
        bits += z.hashCode().toLong()
        return bits.hashCode()
    }


    /**
     * Returns true if all of the data members of Tuple3d t1 are
     * equal to the corresponding data members in this Tuple3d.
     * @param t1  the tuple with which the comparison is made
     * @return  true or false
     */
    fun equals(t1: Tuple3d): Boolean {
        return try {
            x == t1.x && y == t1.y && z == t1.z
        } catch (e2: NullPointerException) {
            false
        }
    }

    /**
     * Returns true if the Object t1 is of type Tuple3d and all of the
     * data members of t1 are equal to the corresponding data members in
     * this Tuple3d.
     * @param other  the Object with which the comparison is made
     * @return  true or false
     */
    override fun equals(other: Any?): Boolean {
        return try {
            val t2 = other as Tuple3d
            x == t2.x && y == t2.y && z == t2.z
        } catch (e1: ClassCastException) {
            false
        } catch (e2: NullPointerException) {
            false
        }
    }

    /**
     * Returns true if the L-infinite distance between this tuple
     * and tuple t1 is less than or equal to the epsilon parameter,
     * otherwise returns false.  The L-infinite
     * distance is equal to MAX[abs(x1-x2), abs(y1-y2), abs(z1-z2)].
     * @param t1  the tuple to be compared to this tuple
     * @param epsilon  the threshold value
     * @return  true or false
     */
    fun epsilonEquals(t1: Tuple3d, epsilon: Float): Boolean {
        var diff: Float = x - t1.x
        if (java.lang.Float.isNaN(diff)) return false
        if ((if (diff < 0) -diff else diff) > epsilon) return false
        diff = y - t1.y
        if (java.lang.Float.isNaN(diff)) return false
        if ((if (diff < 0) -diff else diff) > epsilon) return false
        diff = z - t1.z
        if (java.lang.Float.isNaN(diff)) return false
        return (if (diff < 0) -diff else {
            diff
        }) <= epsilon
    }

    /**
     * Clamps the tuple parameter to the range [low, high] and
     * places the values into this tuple.
     * @param min   the lowest value in the tuple after clamping
     * @param max  the highest value in the tuple after clamping
     * @param t   the source tuple, which will not be modified
     */
    fun clamp(min: Float, max: Float, t: Tuple3d) {
        x = when {
            t.x > max -> {
                max
            }
            t.x < min -> {
                min
            }
            else -> {
                t.x
            }
        }
        y = when {
            t.y > max -> {
                max
            }
            t.y < min -> {
                min
            }
            else -> {
                t.y
            }
        }
        z = when {
            t.z > max -> {
                max
            }
            t.z < min -> {
                min
            }
            else -> {
                t.z
            }
        }
    }

    /**
     * Clamps the minimum value of the tuple parameter to the min
     * parameter and places the values into this tuple.
     * @param min   the lowest value in the tuple after clamping
     * @param t   the source tuple, which will not be modified
     */
    fun clampMin(min: Float, t: Tuple3d) {
        x = if (t.x < min) {
            min
        } else {
            t.x
        }
        y = if (t.y < min) {
            min
        } else {
            t.y
        }
        z = if (t.z < min) {
            min
        } else {
            t.z
        }
    }

    /**
     * Clamps the maximum value of the tuple parameter to the max
     * parameter and places the values into this tuple.
     * @param max the highest value in the tuple after clamping
     * @param t   the source tuple, which will not be modified
     */
    fun clampMax(max: Float, t: Tuple3d) {
        x = if (t.x > max) {
            max
        } else {
            t.x
        }
        y = if (t.y > max) {
            max
        } else {
            t.y
        }
        z = if (t.z > max) {
            max
        } else {
            t.z
        }
    }


    /**
     * Sets each component of the tuple parameter to its absolute
     * value and places the modified values into this tuple.
     * @param t   the source tuple, which will not be modified
     */
    fun absolute(t: Tuple3d) {
        x = Math.abs(t.x)
        y = Math.abs(t.y)
        z = Math.abs(t.z)
    }

    /**
     * Clamps this tuple to the range [low, high].
     * @param min  the lowest value in this tuple after clamping
     * @param max  the highest value in this tuple after clamping
     */
    fun clamp(min: Float, max: Float) {
        if (x > max) {
            x = max
        } else if (x < min) {
            x = min
        }
        if (y > max) {
            y = max
        } else if (y < min) {
            y = min
        }
        if (z > max) {
            z = max
        } else if (z < min) {
            z = min
        }
    }


    /**
     * Clamps the minimum value of this tuple to the min parameter.
     * @param min   the lowest value in this tuple after clamping
     */
    fun clampMin(min: Float) {
        if (x < min) x = min
        if (y < min) y = min
        if (z < min) z = min
    }

    /**
     * Clamps the maximum value of this tuple to the max parameter.
     * @param max   the highest value in the tuple after clamping
     */
    fun clampMax(max: Float) {
        if (x > max) x = max
        if (y > max) y = max
        if (z > max) z = max
    }


    /**
     * Sets each component of this tuple to its absolute value.
     */
    fun absolute() {
        x = abs(x)
        y = abs(y)
        z = abs(z)
    }

    /**
     * Linearly interpolates between tuples t1 and t2 and places the
     * result into this tuple:  this = (1-alpha)*t1 + alpha*t2.
     * @param t1  the first tuple
     * @param t2  the second tuple
     * @param alpha  the alpha interpolation parameter
     */
    fun interpolate(t1: Tuple3d, t2: Tuple3d, alpha: Float) {
        x = (1 - alpha) * t1.x + alpha * t2.x
        y = (1 - alpha) * t1.y + alpha * t2.y
        z = (1 - alpha) * t1.z + alpha * t2.z
    }


    /**
     * Linearly interpolates between this tuple and tuple t1 and
     * places the result into this tuple:  this = (1-alpha)*this + alpha*t1.
     * @param t1  the first tuple
     * @param alpha  the alpha interpolation parameter
     */
    fun interpolate(t1: Tuple3d, alpha: Float) {
        x = (1 - alpha) * x + alpha * t1.x
        y = (1 - alpha) * y + alpha * t1.y
        z = (1 - alpha) * z + alpha * t1.z
    }

    /**
     * Creates a new object of the same class as this object.
     *
     * @return a clone of this instance.
     * @exception OutOfMemoryError if there is not enough memory.
     * @see java.lang.Cloneable
     *
     */
    override fun clone(): Any {
        // Since there are no arrays we can just use Object.clone()
        return try {
            super.clone()
        } catch (e: CloneNotSupportedException) {
            // this shouldn't happen, since we are Cloneable
            throw InternalError()
        }
    }
}