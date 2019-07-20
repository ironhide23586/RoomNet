package com.example.roomnetclassifier;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.opengl.GLSurfaceView;
import android.os.Bundle;

import java.io.IOException;

public class OpenGLActivity extends AppCompatActivity {

    private GLSurfaceView gLView;
    AssetFileDescriptor tfliteModelFd;
//    VisionModel tfliteModel;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        try {
            tfliteModelFd = this.getAssets().openFd("roomnet.tflite");
        }
        catch (IOException e) {
            int k = 0;
        }
        try {
            gLView = new MyGLSurfaceView(this, tfliteModelFd);
        }
        catch (IOException e){
            int k = 0;
        }
        setContentView(gLView);
    }

}
