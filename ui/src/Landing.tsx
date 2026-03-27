import React, { useEffect, useState } from "react";
import "./landing.css";

interface LandingProps {
  onEnterApp: () => void;
}

const DEMO_CONVERSATION = [
  {
    role: "user" as const,
    text: "¿Cómo puedo mejorar la velocidad de mi sitio web?",
  },
  {
    role: "ai" as const,
    text: "Excelente pregunta. Aquí 3 acciones de alto impacto:\n1. Optimiza tus imágenes (usa WebP y lazy-loading)\n2. Activa la caché del navegador y CDN\n3. Minifica tu CSS y JavaScript",
  },
];

export default function Landing({ onEnterApp }: LandingProps) {
  const [scrolled, setScrolled] = useState(false);
  const [demoVisible, setDemoVisible] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setDemoVisible(true), 800);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="landing-root">

      {/* ── NAVBAR ── */}
      <nav className={`ln-nav${scrolled ? " scrolled" : ""}`}>
        <a href="#" className="ln-logo">
          <span className="ln-logo-icon">⚡</span>
          <span className="ln-logo-text">Olympus <span>AI</span></span>
        </a>
        <ul className="ln-links">
          <li><a href="#features">Características</a></li>
          <li><a href="#how">Cómo Funciona</a></li>
          <li><a href="#demo">Demo</a></li>
          <li><a href="#testimonials">Testimonios</a></li>
          <li>
            <a href="#cta" className="ln-nav-cta" onClick={(e) => { e.preventDefault(); onEnterApp(); }}>
              Entrar
            </a>
          </li>
        </ul>
      </nav>

      {/* ── HERO — Patrón Z ── */}
      <section className="ln-hero">
        <div className="ln-hero-columns" />
        <div className="ln-hero-badge">
          ✨ 100% Gratuito · Sin tarjeta de crédito
        </div>
        <h1 className="ln-hero-title">
          La Inteligencia de los <span className="gold">Dioses</span>,<br />
          Al Alcance de Todos.
        </h1>
        <p className="ln-hero-sub">
          Tu asistente de IA personal, gratuito y sin límites ocultos.
          Respuestas claras, rápidas y confiables para cualquier consulta.
        </p>
        <button className="ln-hero-cta" onClick={onEnterApp}>
          ⚡ Comienza Gratis Ahora
        </button>
        <p className="ln-hero-trust">
          Sin registro obligatorio · Sin publicidad · Sin letra pequeña
        </p>
        <div className="ln-hero-scroll">
          <span className="ln-hero-scroll-arrow">↓</span>
          <span>Descubre más</span>
        </div>
      </section>

      {/* ── DIVIDER ── */}
      <div style={{ padding: "0 40px", margin: "0 auto", maxWidth: "1200px" }}>
        <div className="ln-divider" style={{ margin: "48px 0" }}>
          <div className="ln-divider-line" />
          <span className="ln-divider-icon">🏛️</span>
          <div className="ln-divider-line" />
        </div>
      </div>

      {/* ── FEATURES ── */}
      <section id="features" className="ln-section">
        <div className="ln-centered">
          <p className="ln-section-tag">El Poder del Olimpo</p>
          <h2 className="ln-section-title">Todo lo que necesitas, sin costo</h2>
          <p className="ln-section-sub">
            Tres pilares que nos convierten en el asistente que siempre quisiste tener.
          </p>
        </div>
        <div className="ln-features-grid">
          <div className="ln-feat-card">
            <div className="ln-feat-icon">🧠</div>
            <h3 className="ln-feat-title">Inteligencia Real</h3>
            <p className="ln-feat-text">
              Respuestas claras, precisas y sin tecnicismos. Entiende tu pregunta
              en lenguaje natural y te da exactamente lo que necesitas.
            </p>
          </div>
          <div className="ln-feat-card">
            <div className="ln-feat-icon">⚡</div>
            <h3 className="ln-feat-title">Velocidad Divina</h3>
            <p className="ln-feat-text">
              Respuestas en menos de 2 segundos, sin colas de espera ni
              límites de uso diarios. Siempre disponible cuando lo necesitas.
            </p>
          </div>
          <div className="ln-feat-card">
            <div className="ln-feat-icon">🛡️</div>
            <h3 className="ln-feat-title">Privacidad Total</h3>
            <p className="ln-feat-text">
              Tus conversaciones son solo tuyas. No vendemos tus datos,
              no los compartimos y nunca los usamos para publicidad.
            </p>
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <div id="how" className="ln-how-bg">
        <section className="ln-section">
          <div className="ln-centered">
            <p className="ln-section-tag">El Oráculo Habla</p>
            <h2 className="ln-section-title">Tres pasos hacia la sabiduría</h2>
            <p className="ln-section-sub">
              Tan simple como pensar. Sin configuraciones, sin tutoriales, sin complicaciones.
            </p>
          </div>
          <div className="ln-steps">
            <div className="ln-step">
              <div className="ln-step-number">I</div>
              <div className="ln-step-icon">💬</div>
              <h4 className="ln-step-title">Escribe tu pregunta</h4>
              <p className="ln-step-text">
                En español o el idioma que prefieras. No necesitas saber programar
                ni usar términos técnicos.
              </p>
            </div>
            <div className="ln-step">
              <div className="ln-step-number">II</div>
              <div className="ln-step-icon">🔮</div>
              <h4 className="ln-step-title">El Oráculo responde</h4>
              <p className="ln-step-text">
                Nuestros agentes de IA analizan tu consulta y generan
                la respuesta más precisa y útil posible.
              </p>
            </div>
            <div className="ln-step">
              <div className="ln-step-number">III</div>
              <div className="ln-step-icon">🏆</div>
              <h4 className="ln-step-title">Listo para usar</h4>
              <p className="ln-step-text">
                Aplica el conocimiento directamente. Sin copiar, sin traducir,
                sin pasos extra. Información lista para la acción.
              </p>
            </div>
          </div>
        </section>
      </div>

      {/* ── LIVE DEMO ── */}
      <section id="demo" className="ln-section">
        <div className="ln-centered">
          <p className="ln-section-tag">Pruébalo Ahora</p>
          <h2 className="ln-section-title">Ve el Oráculo en acción</h2>
          <p className="ln-section-sub">
            Sin registrarte. Sin esperas. Así responde Olympus AI.
          </p>
        </div>
        <div className="ln-demo-wrapper">
          <div className="ln-demo-header">
            <div className="ln-demo-dots">
              <div className="ln-demo-dot" />
              <div className="ln-demo-dot" />
              <div className="ln-demo-dot" />
            </div>
            <span className="ln-demo-title">⚡ Olympus AI — Conversación demo</span>
          </div>
          <div className="ln-demo-messages">
            {demoVisible && DEMO_CONVERSATION.map((msg, i) => (
              <div key={i} className={`ln-demo-msg ${msg.role}`}>
                <div className="ln-demo-avatar">
                  {msg.role === "ai" ? "⚡" : "👤"}
                </div>
                <div className="ln-demo-bubble">
                  {msg.text.split("\n").map((line, j) => (
                    <span key={j}>{line}<br /></span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="ln-demo-input">
            <span className="ln-demo-input-text">Escribe tu pregunta aquí...</span>
            <button className="ln-demo-send" onClick={onEnterApp}>
              Enviar →
            </button>
          </div>
        </div>
      </section>

      {/* ── TESTIMONIALS ── */}
      <div id="testimonials" className="ln-how-bg">
        <section className="ln-section">
          <div className="ln-centered">
            <p className="ln-section-tag">Lo que dicen los mortales</p>
            <h2 className="ln-section-title">Millones confían en el Olimpo</h2>
          </div>
          <div className="ln-testimonials-grid">
            <div className="ln-testi-card">
              <div className="ln-testi-quote">"</div>
              <p className="ln-testi-text">
                Olympus AI me ahorró 3 horas de trabajo al día. Las respuestas
                son claras y puedo aplicarlas de inmediato sin buscar más.
              </p>
              <div className="ln-testi-stars">★★★★★</div>
              <div className="ln-testi-author">
                <div className="ln-testi-avatar">👩</div>
                <div>
                  <p className="ln-testi-name">María González</p>
                  <p className="ln-testi-role">Diseñadora Freelance</p>
                </div>
              </div>
            </div>
            <div className="ln-testi-card">
              <div className="ln-testi-quote">"</div>
              <p className="ln-testi-text">
                Probé muchos asistentes de IA y ninguno tan rápido ni tan preciso.
                Y sobre todo, ¡gratis! No puedo creer que no cobre nada.
              </p>
              <div className="ln-testi-stars">★★★★★</div>
              <div className="ln-testi-author">
                <div className="ln-testi-avatar">👨</div>
                <div>
                  <p className="ln-testi-name">Carlos Mendoza</p>
                  <p className="ln-testi-role">Emprendedor Digital</p>
                </div>
              </div>
            </div>
            <div className="ln-testi-card">
              <div className="ln-testi-quote">"</div>
              <p className="ln-testi-text">
                Lo uso para resolver dudas de mis clases, resumir textos y hasta
                para planear mis viajes. Es como tener un experto siempre disponible.
              </p>
              <div className="ln-testi-stars">★★★★★</div>
              <div className="ln-testi-author">
                <div className="ln-testi-avatar">🧑</div>
                <div>
                  <p className="ln-testi-name">Andrea López</p>
                  <p className="ln-testi-role">Estudiante Universitaria</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* ── CTA FINAL ── */}
      <section id="cta" className="ln-cta-section">
        <div className="ln-cta-ornament">⚡</div>
        <h2 className="ln-cta-title">
          ¿Listo para consultar al Oráculo?
        </h2>
        <p className="ln-cta-sub">
          Únete a miles de usuarios que ya descubrieron el poder
          de la inteligencia artificial sin barreras.
        </p>
        <button className="ln-cta-btn" onClick={onEnterApp}>
          ⚡ Crear Mi Cuenta Gratis
        </button>
        <p className="ln-cta-note">
          Sin tarjeta de crédito · Acceso inmediato · Para siempre gratis
        </p>
      </section>

      {/* ── FOOTER ── */}
      <footer className="ln-footer">
        <div className="ln-footer-top">
          <div className="ln-footer-brand">
            <a href="#" className="ln-logo">
              <span className="ln-logo-icon" style={{ fontSize: "22px" }}>⚡</span>
              <span className="ln-logo-text" style={{ fontSize: "1.1rem" }}>
                Olympus <span>AI</span>
              </span>
            </a>
            <p>La inteligencia de los dioses, al alcance de todos. Gratis, siempre.</p>
          </div>
          <div className="ln-footer-col">
            <h4>Producto</h4>
            <ul>
              <li><a href="#features">Características</a></li>
              <li><a href="#how">Cómo Funciona</a></li>
              <li><a href="#demo">Demo en Vivo</a></li>
            </ul>
          </div>
          <div className="ln-footer-col">
            <h4>Empresa</h4>
            <ul>
              <li><a href="#">Acerca de</a></li>
              <li><a href="#">Blog</a></li>
              <li><a href="#">Contacto</a></li>
            </ul>
          </div>
          <div className="ln-footer-col">
            <h4>Legal</h4>
            <ul>
              <li><a href="#">Privacidad</a></li>
              <li><a href="#">Términos de Uso</a></li>
              <li><a href="#">Cookies</a></li>
            </ul>
          </div>
        </div>
        <div className="ln-footer-bottom">
          <p>© 2025 Olympus AI · Todos los derechos reservados</p>
          <span className="ln-footer-tagline">Ἡ γνῶσις δύναμίς ἐστιν — El conocimiento es poder</span>
        </div>
      </footer>

    </div>
  );
}
