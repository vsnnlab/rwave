function [new_image] = function_control_shuffle(ORIGINAL,TYPE)
xsize = size(ORIGINAL,2);
ysize = size(ORIGINAL,1);
xcen = xsize/2; ycen = ysize/2;
fft_image = fftshift(fft2(ORIGINAL));
fft_image_1d = fft_image(:);
%% Type 1: orientation map, random generation from isotropic spectrum
if TYPE == 1
% Compute FFT radial spectrum
[xx,yy] = meshgrid(1:xsize,1:ysize);
rr = round(sqrt((xx-xcen).^2+(yy-ycen).^2));
rr_1d = rr(:);
rr_range = min(rr_1d):max(rr_1d);
rr_amp = zeros(size(rr_range));
for r_temp = rr_range
    rr_amp(r_temp) = sqrt(mean(abs(fft_image_1d(rr_1d == r_temp)).^2));
end
rand_phase = 2*pi*gather(gpuArray.rand(size(fft_image)));
rand_phase_1d = rand_phase(:);
new_image_amp = rr_amp(rr_1d);
new_image_real = new_image_amp.*cos(rand_phase_1d)';
new_image_imag = new_image_amp.*sin(rand_phase_1d)';
new_image = (ifft2(ifftshift(reshape(new_image_real+1i*new_image_imag,ysize,xsize))));
end
%% Type 2: activity correlation map, coefficient phase shuffle
if TYPE == 2
fft_amp_1d = abs(fft_image_1d);
rand_phase = 2*pi*gather(gpuArray.rand(size(fft_image)));
rand_phase_1d = rand_phase(:);
new_image_real = fft_amp_1d.*cos(rand_phase_1d);
new_image_imag = fft_amp_1d.*sin(rand_phase_1d);
new_image = (ifft2(ifftshift(reshape(new_image_real+1i*new_image_imag,ysize,xsize))));
end
end