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
%%%%%%%%%% STEREO VISION %%%%%%%%%%
robot.move_j(0,-30,-60,0);
pause(2);
Pose_left = robot.joint_pos;
img_left = getsnapshot(vid);
pause(2);

%robot.move_j(60,-90,-50,-100);
pause(2)
Pose_right = robot.joint_pos;
img_right = getsnapshot(vid);
pause(2);
%%%%%%%%%% RED RECOGNITION %%%%%%%%%%
x_res = 640;
y_res = 480;
frame_middle = [x_res/2,y_res/2];
% detect_bbox_rectangle = [x_res-2,y_res-2,x_res-2,y_res-2];
% detect_bbox = reshape(bbox2points(detect_bbox_rectangle)', 1, []);

% Subtraction red component from grayscale image to extract red
diff_im_left = imsubtract(img_left(:,:,1), rgb2gray(img_left));
diff_im_right = imsubtract(img_right(:,:,1), rgb2gray(img_right));
%Use a median filter to filter out noise
diff_im_left = medfilt2(diff_im_left, [3 3]);
diff_im_right = medfilt2(diff_im_right, [3 3]);
% Convert the resulting grayscale image into a binary image.
diff_im_left = imbinarize(diff_im_left,0.25);
diff_im_right = imbinarize(diff_im_right,0.25);
% Remove all those pixels less than 50px
diff_im_left = bwareaopen(diff_im_left,50);
diff_im_right = bwareaopen(diff_im_right,50);
% Label all the connected components in the image
bw_left = bwlabel(diff_im_left, 8);
bw_right = bwlabel(diff_im_right, 8);
% Image blob analysis
stats_left = regionprops(bw_left, 'BoundingBox', 'Centroid');
stats_right = regionprops(bw_right, 'BoundingBox', 'Centroid');

% Bound red objects in rectangular box
for obj = 1:length(stats_left)
     bbox_left = stats_left.BoundingBox;
     bbox_right = stats_right.BoundingBox;

     if ~isempty(bbox_left) && ~isempty(bbox_right)
         % Display a bounding box around the detected red.
         bboxPoints_left = bbox2points(bbox_left(1, :));
         bboxPolygon_left = reshape(bboxPoints_left', 1, []);
         center_left = [bbox_left(1)+bbox_left(3)/2,bbox_left(2)+bbox_left(4)/2];
         img_left = insertShape(img_left, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
         img_left = insertShape(img_left, 'Line',[frame_middle,center_left],'LineWidth', 5, 'Color',"red");       
         
         bboxPoints_right = bbox2points(bbox_right(1, :));
         bboxPolygon_right = reshape(bboxPoints_right', 1, []);
         center_right = [bbox_right(1)+bbox_right(3)/2,bbox_right(2)+bbox_right(4)/2];
         img_right = insertShape(img_right, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
         img_right = insertShape(img_right, 'Line',[frame_middle,center_right],'LineWidth', 5, 'Color',"red");
     
         distance_center_left = [abs(frame_middle(1)-center_left(1)), abs(frame_middle(2)-center_left(2))]
         distance_center_right = [abs(frame_middle(1)-center_right(1)), abs(frame_middle(2)-center_right(2))]
     end

     % Display the annotated video frame
     %step(videoPlayer, img_left);
     %step(videoPlayer, img_right);
     % Check whether the video player window has been closed.
     if ~isOpen(videoPlayer)
         break
     end
end

% Display stereo images
figure;
subplot(1, 2, 1);
imshow(img_left);
title('Left Image');

subplot(1, 2, 2);
imshow(img_right);
title('Right Image');

% Match features between the two images
pointsLeft = detectSURFFeatures(rgb2gray(img_left));
pointsRight = detectSURFFeatures(rgb2gray(img_right));

[featuresLeft, pointsLeft] = extractFeatures(rgb2gray(img_left), pointsLeft);
[featuresRight, pointsRight] = extractFeatures(rgb2gray(img_right), pointsRight);

indexPairs = matchFeatures(featuresLeft, featuresRight);

matchedPointsLeft = pointsLeft(indexPairs(:, 1));
matchedPointsRight = pointsRight(indexPairs(:, 2));

% Compute the disparity
disparityMap = disparity(rgb2gray(img_left), rgb2gray(img_right));

% Convert disparity to depth
%     Distance: Z = f*B/d 
%     Where f is focal length, B is baseline (c1 c2 distance)
%     d is disparity (horizontal offset of the pointed object)
baseline = abs(Pose_left(1,4)-Pose_right(1,4)); 
focalLength = 1430; 
depthMap = (focalLength * baseline) ./ disparityMap;

% Display depth map
figure;
imshow(depthMap, []);

%%%%%%%%%% CLEAN UP %%%%%%%%%%
while (1)
    if ~isOpen(videoPlayer)
        stop(vid);
        flushdata(vid);
        clear vid;
        clearvars -global
        release(videoPlayer);
        robot.move_j(0,-90,0,0);
        robot.disable_motors();
        clear all;
    end
end
