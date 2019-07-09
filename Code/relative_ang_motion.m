clear all
close all

dt= 0.1;
t = [0:dt:10];
[~,T] = size(t);

phi_ddot = 0.001;

n = 100;
start = 1;
end_x = 100;

l = 0.5;
phi = phi_ddot/2 * t.^2;

x = zeros(n,T);
y = zeros(n,T);

for i = [1:1:n]
    temp_x = zeros(1,T);
    temp_y = zeros(1,T);
    for j = 1:1:i
       temp_x = temp_x + l*cos(phi*j);
       temp_y = temp_y + l*sin(phi*j);
    end
   x(i,:) = temp_x;
   y(i,:) = temp_y;
end

figure();
for time = [1:1:T]
    clf
    plot(x(:,time),y(:,time),'*');
    title(strcat('time =  ',num2str(t(time)) ));
    xlim([0.4,60]);
    ylim([0,35]);
    pause(dt);
end




