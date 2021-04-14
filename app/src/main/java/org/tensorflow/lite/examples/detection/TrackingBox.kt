package org.tensorflow.lite.examples.detection

data class TrackingBox(val frame : Int, val id : Int, val posX : Float, val posY : Float, val width : Float, val height : Float){

    override fun equals(other: Any?): Boolean {
        if(this == other) return true;
        if(javaClass != other?.javaClass) return false;

        other as TrackingBox

        if(frame != other.frame) return false;
        if(id != other.id) return false;
        if(posX != other.posX) return false;
        if(posY != other.posY) return false;
        if(width != other.width) return false;
        if(height != other.height) return false;

        return true;
    }

}

