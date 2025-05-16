const app = document.getElementById('app');

const routes = {
  home: `<section><h2>Bienvenido</h2><p>Explora la TNFR: una ciencia que no representa, sino que reorganiza. Accede a fundamentos, glifos activos y simulaciones resonantes.</p></section>`,
  fundamentos: () => fetch('views/fundamentos.html?v=2').then(res => res.text()),
  glifos: () => fetch('views/glifos.html').then(res => res.text()),
  gpts: () => fetch('views/gpts.html').then(res => res.text()),
  libros: () => fetch('views/libros.html').then(res => res.text()),

};

async function loadPage() {
  const hash = window.location.hash.slice(1) || 'home';
  const content = routes[hash];

  // Cargar el contenido HTML en el <main>
  app.innerHTML = typeof content === 'function' ? await content() : content;

  // Operaciones específicas por sección
  if (hash === 'glifos') {
    const glifos = [
      { nombre: 'A’L', funcion: 'Inicia patrones de resonancia.' },
      { nombre: 'I’L', funcion: 'Estabiliza estructuras coherentes.' },
      { nombre: 'O’Z', funcion: 'Introduce disonancia para reestructurar.' },
      { nombre: 'R’A', funcion: 'Propaga coherencia a nodos vecinos.' }
    ];

    const contenedor = document.getElementById('glifos-lista');
    if (contenedor) {
      glifos.forEach(g => {
        const el = document.createElement('div');
        el.innerHTML = `<strong>${g.nombre}</strong>: ${g.funcion}`;
        el.style.padding = '0.5rem 0';
        contenedor.appendChild(el);
      });
    }
  }
}

// Escuchar cambios en la URL (hash) o carga inicial
window.addEventListener('hashchange', loadPage);
window.addEventListener('load', loadPage);
