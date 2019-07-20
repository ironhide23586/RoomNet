package com.example.roomnetclassifier;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.opengl.GLSurfaceView;

import java.io.IOException;

public class MyGLSurfaceView extends GLSurfaceView {

    private final MyGLRenderer renderer;

    public MyGLSurfaceView(Context context, AssetFileDescriptor fd) throws IOException {
        super(context);

        setEGLContextClientVersion(3);
        renderer = new MyGLRenderer(fd);
        setRenderer(renderer);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY); // redraw iff canvas changed
    }
}
