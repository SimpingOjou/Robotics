%%%%%%%%%% INITIALIZATION %%%%%%%%%%
% Init webcam
vid = videoinput('winvideo', '1', 'MJPG_640x480');
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
pause(2);
%Pose = robot.joint_pos;
img = getsnapshot(vid);
pause(2);

%image 2
robot.move_j(-45,0,-90,0);
pause(2);
%Pose = robot.joint_pos;
img2 = getsnapshot(vid);
pause(2);

%%%%%%%%%% RED RECOGNITION %%%%%%%%%%
x_res = 640;
y_res = 480;
frame_middle = [x_res/2,y_res/2];

% Subtraction red component from grayscale image to extract red
diff_im = imsubtract(img(:,:,1), rgb2gray(img));
%Use a median filter to filter out noise
diff_im = medfilt2(diff_im, [3 3]);
% Convert the resulting grayscale image into a binary image.
diff_im = imbinarize(diff_im,0.25);
% Remove all those pixels less than 50px
diff_im = bwareaopen(diff_im,50);
% Label all the connected components in the image
bw = bwlabel(diff_im, 8);
% Image blob analysis
stats = regionprops(bw, 'BoundingBox', 'Centroid');

% Bound red objects in rectangular box
for obj = 1:length(stats)
     bbox = stats(obj).BoundingBox;

     if ~isempty(bbox)
         % Display a bounding box around the detected red.
         bboxPoints = bbox2points(bbox(1, :));
         bboxPolygon = reshape(bboxPoints', 1, []);
         center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2];
         img = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color',"blue");
         img = insertShape(img, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
         img = insertShape(img, 'Line',[frame_middle,center],'LineWidth', 5, 'Color',"red");

         distance_center = [abs(frame_middle(1)-center(1)), abs(frame_middle(2)-center(2))]
     end
end

% Display stereo images
figure;
imshow(img);
title('Image');

%%%%%%%%%% CLEAN UP %%%%%%%%%%
pause(5);
stop(vid);
flushdata(vid);
clear vid;
clearvars -global
release(videoPlayer);
robot.move_j(0,-90,0,0);
robot.disable_motors();
clear all;
