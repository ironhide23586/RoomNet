package com.example.roomnetclassifier;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.opengl.EGLContext;
import android.opengl.GLSurfaceView;
import android.os.Bundle;

import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import static android.opengl.EGL14.EGL_NO_CONTEXT;
import static android.opengl.EGL14.eglCreateContext;
import static android.opengl.EGL14.eglGetCurrentContext;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glGenBuffers;
import static android.opengl.GLES30.GL_STREAM_COPY;
import static android.opengl.GLES31.GL_SHADER_STORAGE_BUFFER;


public class VisionModel {
    private Activity parent_activity;
//    public String model_fpath;
    public MappedByteBuffer tflite_model;
    public GpuDelegate gpuDelegate;
    public Interpreter tflite_interpreter;
    public Tensor input_tensor, output_tensor;
    public int input_size_bytes = 224 * 224 * 3 * 4;
    public Interpreter.Options tflite_options;

    public Interpreter.Options options;

    public EGLContext eglContext;

    public AssetFileDescriptor tfliteModelFileDescriptor;


    public VisionModel(AssetFileDescriptor model_fd) throws IOException {
        tfliteModelFileDescriptor = model_fd;
//        model_fpath = modelPath;
        tflite_model = loadModelFile();
        tflite_options = new Interpreter.Options();

        gpuDelegate = new GpuDelegate();
        tflite_options.addDelegate(gpuDelegate);
        tflite_interpreter = new Interpreter(tflite_model, tflite_options);

        // Checking for OpenGL context
//        eglContext = eglGetCurrentContext();
////        if (eglContext.equals(EGL_NO_CONTEXT)) return;
//
//        // Creating an OpenGL Shader Storage Buffer Object
//        int[] id = new int[1];
//        glGenBuffers(id.length, id, 0);
//        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
//        glBufferData(GL_SHADER_STORAGE_BUFFER, input_size_bytes, null, GL_STREAM_COPY);
//        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbinding
//        int inSsboId = id[0];
//
//        tflite_interpreter = new Interpreter(tflite_model);
//        input_tensor = tflite_interpreter.getInputTensor(0);
//
//        gpuDelegate = new GpuDelegate();
//        gpuDelegate.bindGlBufferToTensor(input_tensor, inSsboId);
//
//        tflite_interpreter.modifyGraphWithDelegate(gpuDelegate);
//
//        output_tensor = tflite_interpreter.getOutputTensor(0);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
//        AssetFileDescriptor fileDescriptor = parent_activity.getAssets().openFd(model_fpath);
        FileInputStream inputStream = new FileInputStream(tfliteModelFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = tfliteModelFileDescriptor.getStartOffset();
        long declaredLength = tfliteModelFileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer_out = fileChannel.map(FileChannel.MapMode.READ_ONLY,
                startOffset, declaredLength);
        return buffer_out;
    }
}
