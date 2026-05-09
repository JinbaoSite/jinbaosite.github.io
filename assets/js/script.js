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

})();
