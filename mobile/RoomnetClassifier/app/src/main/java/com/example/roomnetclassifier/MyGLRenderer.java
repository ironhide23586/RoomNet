package com.example.roomnetclassifier;

import android.content.res.AssetFileDescriptor;
import android.opengl.EGLContext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
//import android.util.Log;
//import android.widget.Toast;

//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;

//import java.io.FileInputStream;
import java.io.IOException;
//import java.nio.MappedByteBuffer;
//import java.nio.channels.FileChannel;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

//import static android.opengl.EGL14.EGL_NO_CONTEXT;
//import static android.opengl.EGL14.eglGetCurrentContext;


public class MyGLRenderer implements GLSurfaceView.Renderer {

    VisionModel tfliteModel;

    public MyGLRenderer(AssetFileDescriptor fd) throws IOException{
        tfliteModel = new VisionModel(fd);
    }

    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        GLES20.glClearColor(0.0f, 0.0f, 0.5f, 1.0f);
    }

    public void onDrawFrame(GL10 unused) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
    }

    public void onSurfaceChanged(GL10 unused, int width, int height){
        GLES20.glViewport(0, 0, width, height);
    }
}
