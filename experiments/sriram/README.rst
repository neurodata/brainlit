ViterBrain Tracer

Setup:
- Make data is being served with, for example, python cors_webserver.py -d "<path-to-brainlit>/experiments/sriram/data/" -p <port-number>
- Make sure port argument in beginning of trace.py corresponds to <port-number> in above command.
- Make corrections to fragments layer:
    - Source dimensions scale (3um, 510nm, 510nm)
    - Output dimensions names (z,x,y)
- Hide image data from traces layer: uncheck uint16 volume.
- When in middle of trace: option-scroll will expand the skeleton in z.

Tracing workflow tips:
- View only the coronal view window in neuroglancer
- Trace proximally to distally.
- Trace to the end of a process before starting a new branch/neuron.
- Place a point at each branch point (so you can come back and trace the other branch using a hook).

Keyboard Controls - all keys must be paired with SHIFT
- S "select" - place point (either start or end)
- D "draw" - draw line between either the start and end points, or, if no end point exists, start and the cursor
- F "find" - find trace between start and end point via viterbrain 
- Shift+S "save" - save current trace
- Shift+N "new" - new path
- Shift+C "clear" - clear most recent line segment
- Shift+H "hook" - instead of adding the first point for a new trace, you can use this to begin a new branch off an existing trace (it will snap to the closest point of all previously made traces)
