(() => {
  const navHeader = document.querySelector('.nav-header');
  const burger = document.querySelector('.burger');
  const navLinks = document.querySelector('.nav-links');

  if (navHeader && burger && navLinks) {
    burger.addEventListener('click', () => {
      const isOpen = navHeader.classList.toggle('nav-open');
      burger.setAttribute('aria-expanded', String(isOpen));
    });

    navLinks.addEventListener('click', (event) => {
      if (event.target.tagName === 'A' && window.innerWidth < 768) {
        navHeader.classList.remove('nav-open');
        burger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  const backTopButton = document.getElementById('myBtn');
  if (!backTopButton) {
    return;
  }

  const handleScroll = () => {
    const top = Math.max(document.documentElement.scrollTop, document.body.scrollTop);
    backTopButton.style.display = top > 160 ? 'block' : 'none';
  };

  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll();

  backTopButton.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();
