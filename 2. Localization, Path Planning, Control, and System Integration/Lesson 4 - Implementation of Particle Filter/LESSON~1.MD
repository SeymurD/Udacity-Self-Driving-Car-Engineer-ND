# Resources, Hints, and Tips

-   Without implementation of localization methods the car does not know where it is within an acceptable level of precision. The car knows particle coordinates and observation coordinates. The objective is to use the particle coordinates and heading to transform the car's frame of reference to the map's frame of reference, associate the observations, then use the associated observations in the map domain to update the particle weight.
-   Since we know the coordinates of the particle from the car's frame of reference we can use this information and a matrix rotation/translation to transform each observation from the car frame of reference to the map frame of reference. The particle is at (4,5) in the map coordinate system with a heading of -90 degrees. The figure indicates the heading by depicting the particle x-axis as pointing down (blue arrow). This is critical to understanding the matrix transformation we are about to perform.
-   By convention we define the car coordinate system with x pointing up and y rotated from x by pi/2 (+90 degrees). This is another way of saying that y is perpendicular and to the left of x. When x is pointing down, we have the mirror of this, with y perpendicular to the right. To visualize this make an L with your left index finger and thumb with palm facing away from you, this is our map frame of reference. Point the thumb towards the ceiling, this is the car coordinate convention, now point your thumb down, this is the orientation of the particle at (4,5).
-   Now consider the map frame of reference (make an L with your left index finger and thumb as above), this is a typical presentation of Cartesian coordinates, with x pointing right and y perpendicular to the left, pointing up. If we rotate our thumb down we have the particle orientation. To get back to the map orientation we must rotate counterclockwise by 90 degrees (thumb from pointing down back to pointing right). Try this a few times with your left hand. Notice that particle to map is a counterclockwise rotation (+90 degrees) nd map to particle is a clockwise rotation (-90 degrees).
-   The most straight forward way of rotating and translating coordinates is through homogenous transformation matrix (see below) using the angle of rotation required to get to the particle’s frame from the map’s point of view, -90 degrees. This way we can use theta directly. The alternative, which we will not cover here is to is to use -theta and a transformation matrix from the particle frame to the map frame.
-   [This video](https://youtu.be/h11ljFJeaLo)  is a great resource for developing a deeper understanding of how to solve this transformation problem - it covers the rotation transformation, and from there you just need to perform a translation.

Observations in the car coordinate system can be transformed into map coordinates ($x_m$​  and  $y_m$​) by passing car observation coordinates ($x_c$​  and  $y_c$​), map particle coordinates ($x_p$ and $y_p$​), and our rotation angle (-90 degrees) through a homogenous transformation matrix. This homogenous transformation matrix, shown below, performs rotation and translation.


$$
\left(\begin{array}{cc}
		x_m \\ y_m \\ 1 
	\end{array}\right) = 
\left(\begin{array}{cc}
	cos(\theta) & -sin(\theta) & x_p \\
	sin(\theta) & cos(\theta) & y_p \\
	0 & 0 & 1
\end{array}\right)
\left(\begin{array}{cc}
	x_c \\ y_c \\ 1 
\end{array}\right)
$$ 

