## Swinging Search and Crawling Control

## Structure
1. `arm_kine`: my manipulator kinematics package.
   - forward kinematics: Given angles of the arm, calculate the location of the end.
   - Jacobian: Calculate the Jacobian matrix for inverse kinematics.
   - inverse kinematics: Given the velocity of the end, calculate the velocity of the angles.
2. `coding_steps`: steps 1-4 are runnable.
   - step1: Test the forward kinematics and the inverse kinematics. Control the end to move as desired.
   - step2: Control the end to move to the target position, and then start self-motion.
   - step3: Generate target end positions in the bucket for the task.
   - step4: Swinging search randomly, to find a collision-free configuration.
   - step5-9: Swinging search by DDPG. Some codes were left in a lab server and has been lost.
3. `crawling`: Crawling Control.
All the videos and the pics in the paper can be reproduced by running these codes.

![Crawling Control](./video/simple_16_2_speedup.mp4)