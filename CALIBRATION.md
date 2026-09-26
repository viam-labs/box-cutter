Guide on how to calibrate the box bot (an arm equipped with a camera and a cutting tool).
Step 1: home position and the camera 
Choose a home position and park the arm with the camera there. Then, physically measure the distance from the camera lens to the top of the box - this is "depth_mm", the first parameter of the box presets. This parameter configures how low does the arm descent when it moves to the center of the box. 
Step 2: find the u,v pixels of the center of the box. This can be done in the camera control tab on the Viam app. Simply hover over the center of the box with the mouse coordinate on setting, and record the first value for u, and the second for v (2nd and 3rd parameters of the box presets). 
Step 3: saving the box parameters. Write down the values above in the preset field of the module, and then use the "set_box" do command  to apply the preset. "set_box" also allows for fully manual box setting, where the user can supply all 5 required values for the box right from the viam app.
Step 4: calculating the distance from the base of the arm to the closer edge of the box. Changaeble by "stopper_x_mm" config attribute - this distance is needed for accurate side seam stagings. An easy way to check it is to comment out the last 2 out of 3 moves in "stage_side_seam" for the close seam - the arm will stop at the stopper_x_mm value. 
Step 5: cutting adjustments. These is the most experimental part, since it works with the blade. Depending on the "depth_mm" and "stopper_x_mm, the blade can be inserted too far or not enough into the box. The user can manipulate either the values above, or the insert_blade_mm config attribute, but it works slightly differently for top and side seams. 

Running the module: 
- go to home
- run set_box with either a preset or custom dimensions
- run move_to_center
- run convergence, and check that the blade is above the top seam
- run cut or cut, "seam" : "top" (both should resolve) 
- move_to_seam far
- converge
- cut 
- move to seam close
- converge
- cut close
- home

If the whole process is working, running "full_cut" does all of the above steps (but the user should still set the box before).
Jot something down
