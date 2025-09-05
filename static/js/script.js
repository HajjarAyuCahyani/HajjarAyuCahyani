let currentIndex = 0;
const slides = document.querySelectorAll('.carousel-slide');

function showSlide(index) {
    slides.forEach((slide, i) => {
        slide.classList.remove('active');
        if (i === index) slide.classList.add('active');
    });
}

function changeSlide(direction) {
    currentIndex += direction;
    if (currentIndex < 0) currentIndex = slides.length - 1;
    if (currentIndex >= slides.length) currentIndex = 0;
    showSlide(currentIndex);
}

// Inisialisasi pertama
if (slides.length > 0) {
    showSlide(currentIndex);
}
