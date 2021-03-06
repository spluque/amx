% amxLoad
% loads AMX files into matrices
% modified 2/7/2018

% Surface pressure estimate in mbar for depth calculation
% depth calculation assumes 1 bar = 10 m
surfacepress=1010;

% %% Comment to turn off batch loading
% PathName=uigetdir();
% cd(PathName);
% files=dir('*.AMX');
% nfiles=length(files);
%%

%% Uncomment to load one file at a time using dialog box
[FileName,PathName,FilterIndex] = uigetfile({'*.amx','AMX files (*.amx)'},'Select an AMX file');
if isequal(FileName,0)|isequal(PathName,0)
   return
end
nfiles = 1;
files(1).name = FileName

%%

cd(PathName);

AUDIO=[];
PT=[];
RGB=[];
IMU=[];
O2=[];

INER=[];
INER.accel=[];INER.mag=[];INER.gyro=[];
ADC=[];
PTMP=[];
INER_ts=[];
PTMP_ts=[];
light = [];
                
for n=1:nfiles
    files(n).name % print filename
    %% Load file
    [DF_HEAD, SID_SPEC, SID_REC]=oAMX(files(n).name);

    for x=1:length(SID_REC)
        cur_sid=(SID_REC(x).nSID) + 1;
        if(SID_SPEC(cur_sid).SID(1)=='A')
            AUDIO=vertcat(AUDIO,SID_REC(x).data);
        end
        if(SID_SPEC(cur_sid).SID(1)=='P')
            PT=vertcat(PT,SID_REC(x).data);
        end
        if(SID_SPEC(cur_sid).SID(1)=='L')
            RGB=vertcat(RGB,SID_REC(x).data);
            RGB_SID = cur_sid;
        end
        if(SID_SPEC(cur_sid).SID(1)=='I' | SID_SPEC(cur_sid).SID(1)=='3')
            IMU=vertcat(IMU,SID_REC(x).data);
            IMU_SID = cur_sid;
        end
            if(SID_SPEC(cur_sid).SID(1)=='O')
            O2=vertcat(O2,SID_REC(x).data);
        end
    end
end

% Split IMU and RGB into structure
INER.accel.x = [IMU(1:9:end) * SID_SPEC(IMU_SID).sensor.cal(1)];
INER.accel.y = [IMU(2:9:end) * SID_SPEC(IMU_SID).sensor.cal(2)];
INER.accel.z = [IMU(3:9:end) * SID_SPEC(IMU_SID).sensor.cal(3)];

INER.gyro.x = [IMU(4:9:end) * SID_SPEC(IMU_SID).sensor.cal(4)];
INER.gyro.y = [IMU(5:9:end) * SID_SPEC(IMU_SID).sensor.cal(5)];
INER.gyro.z = [IMU(6:9:end) * SID_SPEC(IMU_SID).sensor.cal(6)];

INER.mag.x = [IMU(7:9:end) * SID_SPEC(IMU_SID).sensor.cal(7)];
INER.mag.y = [IMU(8:9:end) * SID_SPEC(IMU_SID).sensor.cal(8)];
INER.mag.z = [IMU(9:9:end) * SID_SPEC(IMU_SID).sensor.cal(9)];

light.red = [RGB(1:3:end) * SID_SPEC(RGB_SID).sensor.cal(1)];
light.green = [RGB(2:3:end) * SID_SPEC(RGB_SID).sensor.cal(2)];
light.blue = [RGB(3:3:end) * SID_SPEC(RGB_SID).sensor.cal(3)];