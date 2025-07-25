# Florine's contribution to Cocofest data processing

## Force c3d files processing

Python file : `c3d_to_force.py` \
Class : `C3dToForce` \
This class allows to :
1. read c3d file : `_load_c3d()`
2. filter data
3. calibrate data (V to N) : `_calibration()`
4. set zero level (cancel gravity effects) : `set_zero_level()`
5. get stimulations times and indexes from c3d data : `get_stimulation()`
6. slice data to separate each train (easier for identification) : `slice_data()`
7. set to zero the "rest" part (for better identificaion optimisation) : `set_to_zero_slice()`

All these steps allows to get the force data at the handle `get_data_at_handle()`. To get force data at the muscle, you can either use the functions of the C3dToForce class or the static optimisation developped in `static_biosiglive.py`.

In the C3dToForce class, you can :
1. change the frame of reference from local sensor (handle) to local hand one : `local_sensor_to_local_hand()`
2. select the muscle and degree of freedom you want : `select_muscle_and_dof()`. This method is accurate only for one muscle, if you want to select more than one muscle, you must use the static optimisation method.
3. get the muscle force from local data : `get_muscle_force()`
4. save or plot the results : `get_force(plot, save)`

Using the biosiglive static optimisation (`static_biosiglive.py`), you can :
1. get the muscles force from data at the handle usg the class MskFunctions : `get_muscles_forces()`
2. use the automatic function to compute all participants data : `auto_process()`, you have to specify the participant number and if you want to plot or save the results
3. check the results by ploting them : `check_data()`, you only have to specify the participant number you want

## Motion c3d files processing

Python fie : `c3d_to_q.py` \
Class : `C3dToQ` \
This class allows to :
1. read c3d file : `load_c3d()` and `load_analog()`
2. get segment's vector from markers' positions : `_get_segment_vectors()`
3. project vectors on the same plane : `_projection_vectors()`
4. get the angle between the projected vectors : `_get_angle()`
5. choose to get the angle either in degrees or radians : `get_q_deg()` or `get_q_rad()`
6. slice data to separate each train (easier for identification) : `slice_data()`  
7. get the sliced data : `get_sliced_time_Q_deg()` or `get_sliced_time_Q_rad()`