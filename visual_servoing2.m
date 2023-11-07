a = imaqhwinfo;
% [camera_name, camera_id, format] = getCameraInfo(a);
% Capture the video frames using the videoinput function
% You have to replace the resolution & your installed adaptor name.
vid = videoinput('winvideo', '1');
% Set the properties of the video object
set(vid, 'FramesPerTrigger', Inf);
set(vid, 'ReturnedColorspace', 'rgb')
vid.FrameGrabInterval = 1;
%start the video aquisition here
start(vid)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_res = 640;
y_res = 480;
frame_middle = [x_res/2,y_res/2];
detect_bbox_rectangle = [x_res-2,y_res-2,x_res-2,y_res-2];
detect_bbox = reshape(bbox2points(detect_bbox_rectangle)', 1, []);

videoPlayer = vision.VideoPlayer


%Init Robot
robot = MyRobot();
assert(robot.is_robot_connected(),"Robot not connected properly");
robot.move_j(0,-90,0,0);
cw = 0;
pause(2);

while true  
        disp('here')
    % % % %         videoFrame = snapshot(cam);
    % % % %         videoFrameGray = rgb2gray(videoFrame);
    % % % %         %bbox = faceDetector.step(videoFrameGray,face_detect_bbox_rectangle);
    % % % %         bbox = faceDetector.step(videoFrame);
                    % Get the snapshot of the current frame
        data = getsnapshot(vid);
        data2 = data;
    
        %%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%
        
        % Now to track red objects in real time
        % we have to subtract the red component 
        % from the grayscale image to extract the red components in the image.
        diff_im = imsubtract(data(:,:,1), rgb2gray(data));
        %Use a median filter to filter out noise
        diff_im = medfilt2(diff_im, [3 3]);
        % Convert the resulting grayscale image into a binary image.
        diff_im = im2bw(diff_im,0.25);
        
        % Remove all those pixels less than 300px
        diff_im = bwareaopen(diff_im,300);
        
        % Label all the connected components in the image.
        bw = bwlabel(diff_im, 8);
        
        % Here we do the image blob analysis.
        % We get a set of properties for each labeled region.
        stats = regionprops(bw, 'BoundingBox', 'Centroid');
        
        % Display the image
        %imshow(data)
        
        % hold on

                    %This is a loop to bound the red objects in a rectangular box.
            for object = 1:length(stats)
                bb = stats(object).BoundingBox;
                bc = stats(object).Centroid;
                rectangle('Position',bb,'EdgeColor','r','LineWidth',2)
                plot(bc(1),bc(2), '-m+')
                a=text(bc(1)+15,bc(2), strcat('X: ', num2str(round(bc(1))), '    Y: ', num2str(round(bc(2)))));
                set(a, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'yellow');
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                bbox = bb
                bboxPoints = bbox2points(bbox(1, :));
        
                % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
                % format required by insertShape.
        
                bboxPolygon = reshape(bboxPoints', 1, []);
                center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2];
        
                detect_bbox_color = "green";
                dists = get_distances(center, frame_middle);
                % Display a bounding box around the detected face.
                data2 = insertShape(data2, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color',"blue");
                data2 = insertShape(data2, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
                data2 = insertShape(data2, 'Line',[frame_middle,center],'LineWidth', 5, 'Color',"red");
                delta_j1 = (frame_middle(1)-center(1))/100;
                delta_j4 = (frame_middle(2)-center(2))/100;
        
        
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             
        
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
                try
                    robot.move_j(robot.joint_angles(1)+delta_j1,robot.joint_angles(2),robot.joint_angles(3),robot.joint_angles(4));
                    robot.move_j(robot.joint_angles(1),robot.joint_angles(2),robot.joint_angles(3)+delta_j4,robot.joint_angles(4));
    
                catch ME
                    disp(ME.message);
                end
            %
            end

            % Display the annotated video frame using the video player object.
            
    % % %     step(videoPlayer, videoFrame);
        step(videoPlayer, data2);
    
        % Check whether the video player window has been closed.
        runLoop = isOpen(videoPlayer);
        if ~runLoop
            break
        end

        flushdata(vid);
end

stop(vid);
flushdata(vid);
clearvars -global
% Clean up.
release(videoPlayer);
release(faceDetector);
robot.disable_motors();



function cw = jog_robot_z(cw,robot)
    if robot.joint_angles(1) == -130
        cw = 1;
    elseif robot.joint_angles(1) == 130
        cw = 0; 
    end
    
    if ~cw
       robot.move_j(robot.joint_angles(1)-10,robot.joint_angles(2),robot.joint_angles(3),robot.joint_angles(4));
    else
        robot.move_j(robot.joint_angles(1)+10,robot.joint_angles(2),robot.joint_angles(3),robot.joint_angles(4));
    end

end


function dists = get_distances(center, frame_middle)
    x_dist = frame_middle(1) - center(1);
    y_dist = frame_middle(2) - center(2);
    dists = [x_dist,y_dist];
end

function center = get_center(bboxPolygon)
    x = floor((bboxPolygon(1) + bboxPolygon(3)) /2);
    y = floor((bboxPolygon(2) + bboxPolygon(4)) /2);
    center = [x,y];

end

function inside = face_inside_bbox(center, frame_middle)
    if abs(frame_middle(1) - center(1)) < frame_middle(1) && abs(frame_middle(2) - center(2)) < frame_middle(2)
        inside = 1;
    else
        inside = 0;
    end
end

