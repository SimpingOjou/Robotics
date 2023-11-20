%%%%%%%%%% INITIALIZATION %%%%%%%%%%
% Init webcam
vid = videoinput('winvideo', '1', 'MJPG_1280x720');
set(vid, 'FramesPerTrigger', Inf);
set(vid, 'ReturnedColorspace', 'rgb');
vid.FrameGrabInterval = 2;
start(vid);
videoPlayer = vision.VideoPlayer;

% Init Robot
robot = MyRobot();
assert(robot.is_robot_connected(),"Robot not connected properly");

%%%%%%%%%% IMAGE ACQUISITION %%%%%%%%%%
robot.move_j(0,0,-90,0);
pause(5);
img = getsnapshot(vid);
pause(2);

%%%%%%%%%% RED RECOGNITION %%%%%%%%%%
x_res = 1280;
y_res = 720;
frame_middle = [x_res/2,y_res/2];

% Subtraction red component from grayscale image to extract red
diff_im = imsubtract(img(:,:,1), rgb2gray(img));
%Use a median filter to filter out noise
diff_im = medfilt2(diff_im, [3 3]);
% Convert the resulting grayscale image into a binary image.
diff_im = imbinarize(diff_im, 0.15);
% Remove all those pixels less than 50px
diff_im = bwareaopen(diff_im,50);
% Label all the connected components in the image
bw = bwlabel(diff_im, 8);
% Image blob analysis
stats = regionprops(bw, 'BoundingBox', 'Centroid');

% Bound red objects in rectangular box
[x,y,z] = robot.read_ee_position(); % reads position in cartesian
for obj = 1:length(stats)
     bbox = stats(obj).BoundingBox;

     if ~isempty(bbox)
         % Display a bounding box around the detected red.
         bboxPoints = bbox2points(bbox(1, :));
         bboxPolygon = reshape(bboxPoints', 1, []);
         obj_size_px = [bbox(1)+bbox(3), bbox(2)+bbox(4)];

         img = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color',"blue");
         img = insertShape(img, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
         img = insertShape(img, 'Line',[frame_middle,center],'LineWidth', 5, 'Color',"red");       
     
         distance_center_px = [abs(frame_middle(1)-(obj_size_px(1)/2)), abs(frame_middle(2)-(obj_size_px(2)/2))]
         
         %%%%%%%%%% PIXEL TO REAL WORLD CONVERSION %%%%%%%%%%
         % focal_length_px = 1430; % pixels
         % camera_height = 0.125; % meters
         % fov = 60 % degrees

         px_m_ratio = (10,5) % whatever we find with the ruler on x and y
         obj_distance_m = [distance_center_px(1)*px_m_ratio(1), distance_center_px(2)*px_m_ratio(2)]

         %%%%%%%%%% INVERSE KINEMATICS %%%%%%%%%%
         robot.move_c(x+obj_distance_m(1),y+obj_distance_m(2),z,0); % check xyz coordinates with robot.draw()
         % In the line above I assumed X is horizontal, Y is vertical and Z
         % is depth
         pause(3);
         robot.move_j(0,0,-90,0); % goes back to place
         pause(2);
     end
end


% Display stereo images
figure;
imshow(img);
title('Image');

%%%%%%%%%% CLEAN UP %%%%%%%%%%
pause(3);
stop(vid);
flushdata(vid);
clear vid;
clearvars -global
release(videoPlayer);
robot.move_j(0,-90,0,0);
robot.disable_motors();
clear all;