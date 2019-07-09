clear all
close all

a1 = 1;
a2 = 2;
l = 1;
t = [0:0.1:10];
z = [0.25,0.5,0.75];


for i = 1:1:3
    theta1(:,i) = (a1*(t - 2*pi*z(i)/l + 2*pi*z(1))) .* heaviside(t - 2*pi*z(i)/(l) + 2*pi*z(1));
    theta2(:,i) = (a2*(t - 2*pi*z(i)/l + 2*pi*z(1))) .* heaviside(t - 2*pi*z(i)/(l) + 2*pi*z(1));
end

figure()
plot(t,theta1)
title('Roll angle at each segment at A = 1');
xlabel('time(s)');
ylabel('Angle');
legend('z=0.25','z=0.5','z=0.75');
ylim([0,20]);
figure()
plot(t,theta2)
title('Roll angle at each segment at A = 2');
xlabel('time(s)');
ylabel('Angle');
legend('z=0.25','z=0.5','z=0.75');