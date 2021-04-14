package org.tensorflow.lite.examples.detection.point

class PlaneMgr {
    private val planeSet = mutableListOf<Plane>()
    fun add(depth: Float, points: Array<Pair<Point2d, Point3d>>){
        val plane = Plane(depth)
        for (point in points){
            plane.add(point.first, point.second)
        }
        planeSet.add(plane)
    }
    fun sort(){
        planeSet.sortBy { it.depth }
    }
    fun search(point2d: Point2d): Plane?{
        var plane: Plane? = null
        for (candPlane in planeSet){
            if(isInside(point2d, candPlane)){
                plane = candPlane
                break
            }
        }
        return plane
    }
    fun baseDepth():Float{
        return planeSet.first().depth
    }
    fun getDepthList():FloatArray{
        val depthList = FloatArray(planeSet.size)
        for (i in 0 until planeSet.size){
            depthList[i] = planeSet[i].depth
        }
        return depthList
    }
    private fun isInside(point2d: Point2d, plane: Plane): Boolean{
        return plane.isInside(point2d)
    }
    class Plane(val depth:Float){
        private val vertices = mutableListOf<Pair<Point2d, Point3d>>()
        fun add(point2d: Point2d, point3d: Point3d){
            vertices.add(Pair(point2d, point3d))
        }
        fun isInside(point2d: Point2d): Boolean{
            var crosses:Boolean = false
            var j = vertices.size -1
            for (i in 0 until vertices.size){
                if (((vertices[i].first.y > point2d.y) != (vertices[j].first.y > point2d.y)) && (point2d.x < ((vertices[j].first.x - vertices[i].first.x) * (point2d.y - vertices[i].first.y) / (vertices[j].first.y - vertices[i].first.y) + vertices[i].first.x))){
                    crosses = !crosses
                }
                j = i
            }
            return crosses
        }
    }
}